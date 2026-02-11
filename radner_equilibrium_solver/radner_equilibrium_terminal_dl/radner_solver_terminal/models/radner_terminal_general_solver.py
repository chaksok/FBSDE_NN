import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import time
from abc import ABC,abstractmethod
from torch.func import jacrev, vmap

from core.network import FeedForwardNN as myNN
from core.data_generator import BrownianMotionGenerator
from core.base_solver import FBSDEBase


class TerminalSolver(FBSDEBase, ABC):
    """Abstract base class for FBSDE solvers using neural networks.
    
    This class provides the training loop, validation, and checkpointing.
    Subclasses must implement problem-specific methods (drift, volatility, etc.).
    """
    
    def __init__(self, config):
        """Initialize solver with configuration dictionary.
        
        Args:
            config: Dictionary containing:
                - T: Terminal time
                - M: Batch size (number of trajectories)
                - N: Number of time steps
                - D: State dimension
                - I: Number of agents
                - K: Validation set size
                - learning_rate: Learning rate for Adam
                - device: 'cpu' or 'cuda'
                - checkpoint_path: Path to save best model
        """
        super().__init__(config)
        self.drift_D = config['drift_D']   ## coef of D=g(t,X) with respect to t
        self.sigD = config['sigD'].to(self.device)  ##coef of D=g(t,X) with respect to X
        self.muE = config['muE'].to(self.device)    ## coef of E=f_i(t,X) with respect to t
        self.sigE = config['sigE'].to(self.device)   ## coef of E=f_i(t,X) with respect to X
        self.alpha = config['alpha']
        self.epsilon = config['epsilon']


    def dividend_process(self, t, X):  #dividend process Di=g(t,X)
        """Compute the dividend process."""
        mu = self.drift_D
        sig = self.sigD
        if len(sig[0, :]) != self.D:
            raise ValueError(f"the volatility of the asset should have {self.D} elements.\n")
        return torch.sum(sig * X, -1, keepdims=True)
    

    def dividend_gradient(self,t,x):  ## first order derivative with respect to t and X
        """
        Compute the first order derivative of the dividend process Di = g(t, X) with respect to t and X.

        Parameters:
            t (torch.Tensor): time
            x (torch.Tensor): state process X

        Returns:
            torch.Tensor: derivative of the dividend process with respect to t and X, of shape (M, D)
        """
        x.requires_grad=True
        gDx=torch.autograd.grad(self.dividend_process(t,x), x, grad_outputs=torch.ones_like(self.dividend_process(t,x)))[0]
        x.requires_grad=False
        return gDx
    
    
    def endowment_process(self,t,X,i): #endowment process for agent i, E^i=g^i(t,X)
        """
        Compute the endowment process E^i=g^i(t,X) for agent i.
        """
        mu=self.muE
        sig=self.sigE
        A = 1.0 
        if len(mu)!=self.I:
            raise ValueError(f"the elements in the drift term of the endowments do not match the number of agents {self.I}.\n")
        if len(sig)!=self.I:
            raise ValueError(f"there should be {self.I} agents in the market.\n")
        if len(sig[0,:])!=self.D:
            raise ValueError(f"the volatility of endowments should be {self.D} dimension.\n")
        
        temp = torch.sum(sig[i,:]*X,-1,keepdims=True)
        if i == 0: 
            return -A * torch.minimum(temp, torch.zeros_like(temp))
        else: 

            return A/(self.I-1) * torch.maximum(temp, torch.zeros_like(temp))
    
    def endowment_gradient(self,t,x,i): ## first order derivative with respect to t and X
        """
        Compute the first order derivative of the endowment process E^i=g^i(t,X) for agent i with respect to t and X.

        Parameters:
            t (torch.Tensor): time
            x (torch.Tensor): state process X
            i (int): agent index

        Returns:
            torch.Tensor: derivative of the endowment process with respect to t and X, of shape (M, D)
        """
        x.requires_grad=True
        gEx=torch.autograd.grad(self.endowment_process(t,x,i), x, grad_outputs=torch.ones_like(self.endowment_process(t,x,i)))[0]
        x.requires_grad=False
        return gEx
    
    
    def phi(self,t,X,Z,i,epsilon): #i is the order of the agent (start from 1), epsilon is the error we can bear in the computation 
        """
        Compute the phi term in the differential equation.

        Parameters:
            t (torch.Tensor): time
            X (torch.Tensor): state process X
            Z (torch.Tensor): Brownian motion
            i (int): order of the agent (start from 1)
            epsilon (float): error we can bear in the computation

        Returns:
            torch.Tensor: phi term, of shape (M, D)
        """
        I=self.I
        alpha=self.alpha
        D=self.D
        Z0=Z[:,0:D]
        
        
        part2=0.5*torch.sum(Z[:,i*D:(i+1)*D]**2,-1,keepdims=True)
        if torch.norm(Z0)<epsilon:
            return part2
        else:
            unit=Z0/torch.sqrt(torch.sum(Z0**2,-1,keepdims=True))
            temp=Z0
            for j in range(I):
                temp=temp+alpha[j]*Z[:,(j+1)*D:(j+2)*D]
            part1=torch.sum((temp-Z[:,i*D:(i+1)*D])*unit,-1,keepdims=True)
            return part2-0.5*part1**2  ##M*1
    
    
    def phi0(self,t,X,Z,epsilon): #D:M*1
        """
        Compute the phi0 term in the differential equation.
        
        Parameters:
            t (torch.Tensor): time
            X (torch.Tensor): state process X
            Z (torch.Tensor): Brownian motion
            epsilon (float): error we can bear in the computation
        
        Returns:
            torch.Tensor: phi0 term, of shape (M, D)
        """
        I=self.I
        alpha=self.alpha
        D=self.D
        
        Z0=Z[:,0:D]
        temp=Z0
        for i in range(I):
            temp=temp+alpha[i]*Z[:,(i+1)*D:(i+2)*D]
        
        return torch.sum(temp*Z0,-1,keepdims=True)  #M*1
        
    
    def mu(self,t,X):  ## drift of state process X
        """
        Compute the drift term of the state process X.
        
        Parameters:
            t (torch.Tensor): time
            X (torch.Tensor): state process X
        
        Returns:
            torch.Tensor: drift term, of shape (M, D)
        """
        return super().mu(t,X)
             
    def sig(self,t,X): #M*1,M*D,M*1, volatility of state process X
        """
        Compute the volatility matrix of the state process X.
        
        Parameters:
            t (torch.Tensor): time
            X (torch.Tensor): state process X
        
        Returns:
            torch.Tensor: volatility matrix, of shape (M, D, D)
        """
        size = X.shape
        a = torch.ones(size).to(self.device)
        return torch.diag_embed(a) #M*D*D
     
    
    def theta(self,Y,Z,K):  ## portfolio process
        """
        Compute the portfolio process θ(t,W) for K agents.

        Parameters:
            Y (torch.Tensor): not used
            Z (torch.Tensor): Brownian motion
            K (int): number of agents

        Returns:
            list of torch.Tensor: portfolio process for each agent, of shape (K, I)
        """
        I = self.I
        Z0 = Z[:,0:self.D]
        
        theta = []
        if torch.norm(Z0)<1e-7:
            temp = torch.randn(K,I).to(self.device)
            temp /= torch.sum(temp, -1, keepdims=True)
            for i in range(I):
                theta.append(temp[:,i].unsqueeze(-1))
            return theta
        else:
            temp = Z0 / torch.sum(Z0**2, -1, keepdims=True) #K*D
            tmp = 0
            for i in range(I):
                tmp = tmp + self.alpha[i] * Z[:,(i + 1) * self.D : (i+2) * self.D]

            for i in range(I):
                theta_i = 1 + torch.sum((tmp-Z[:,(i + 1) * self.D : (i+2) * self.D]) * temp, -1, keepdims=True)
                theta.append(theta_i)
    
            return theta
    


torch.manual_seed(1)
np.random.seed(1)

    
class RadnerEquilibriumSolver(TerminalSolver):
    """Solver for Radner equilibrium problems.
    
    This implements the specific dynamics and terminal conditions for
    a multi-agent Radner equilibrium with CARA utility.
    """
    
    def __init__(self, config):
        """Initialize Radner equilibrium solver.
        
        Args:
            config: Configuration dictionary (see FBSDESolver)
            drift_D: Drift coefficient for dividend process
            sigD: Volatility coefficients for dividend (1 x D tensor)
            muE: Drift coefficients for endowments (I-length tensor)
            sigE: Volatility coefficients for endowments (I x D tensor)
            alpha: Agent risk-aversion coefficients (I-length list)
            epsilon: Numerical tolerance
        """
        super().__init__(config)    
        self.layers_z = config['layers_z']   
        self.layers_y = config['layers_y']

        # Neural network and optimizer
        self.model_z = myNN(self.layers_z).to(self.device)

        self.model_y0 = myNN(self.layers_y).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.model_z.parameters()) + list(self.model_y0.parameters()),
            lr=self.learning_rate
        )

        # Test data for prediction
        self.t_star,self.W_star = BrownianMotionGenerator.generate(self.K, self.N, self.D, self.T)
    
    
    def net_z(self,t,x): 
        """
        Compute the neural network output z(t, x) and its gradient with respect to x.

        Args:
            t (torch.Tensor): Time of shape (M, 1)
            x (torch.Tensor): State of shape (M, D)

        Returns:
            z (torch.Tensor): Neural network output of shape (M, Ii*D)
        """
        x.requires_grad = True
        z=self.model_z(t, x)
        return z
    

    def net_y(self,t,x): 
        """
        Compute the neural network output y0(t, x) and its gradient with respect to x.

        Args:
            t (torch.Tensor): Time of shape (M, 1)
            x (torch.Tensor): State of shape (M, D)

        Returns:
            y0 (torch.Tensor): Neural network output of shape (M, I*1)
        """
        x.requires_grad = True
        y0=self.model_y0(t, x)
        return y0


    def loss_func(self, t, W): #Xi is the initial X at time 0
        """
        Compute the loss function for the Radner equilibrium problem.

        Args:
            t (torch.Tensor): Time of shape (M, N+1)
            W (torch.Tensor): Brownian motion of shape (M, N+1, D)

        Returns:
            loss (torch.Tensor): Loss tensor of shape (1)
        """
        I=self.I
        loss=torch.tensor([0.0],device = self.device)
  
        t0=t[:,0,:].float() #M*1 initial time
        W0=W[:,0,:].float() #M*D

        Y0=self.net_y(t0,W0)  #M*(I*1), M*(I*D)
        Z0 = self.net_z(t0, W0)
        for i in range(I+1):
            Z0[:,i*self.D:(i+1)*self.D]=(Z0[:,i*self.D:(i+1)*self.D].unsqueeze(1)@self.sig(t0,W0)).squeeze()


        for time in range(1,self.N+1): #iterate from time 1 to time N
            t1=t[:,time,:].float() #M*1
            W1=W[:,time,:].float() #M*1
            
            Z0_0=Z0[:,0:self.D]
            
            Y1_temp=[]
            Y0_1_pred = Y0[:,0].unsqueeze(1) + self.phi0(t0,W0,Z0,self.epsilon)*(t1-t0)+torch.sum(Z0_0*(W1-W0),1,keepdims=True)
            Y1_temp.append(Y0_1_pred)
            for j in range(1,I+1):
                pred=Y0[:,j].unsqueeze(1)+self.phi(t0,W0,Z0,j,self.epsilon)*(t1-t0)+torch.sum(Z0[:,j*self.D:(j+1)*self.D]*(W1-W0),1,keepdims=True)
                Y1_temp.append(pred)
            
            Y1=torch.cat(Y1_temp,1)  #M*(I+2)
            
            Z1=self.net_z(t1,W1)
            for i in range(I+1):
                Z1[:,i*self.D:(i+1)*self.D]=(Z1[:,i*self.D:(i+1)*self.D].unsqueeze(1)@self.sig(t1,W1)).squeeze()
        
            
            t0, W0, Y0, Z0 = t1, W1, Y1, Z1
                   
        Di_0 = self.dividend_process(t1,W1) #M*1
        E_0 = [self.endowment_process(t1,W1,i) for i in range(I)]  #list of M*1
        
        Y_terminal_temp = []
        Y_terminal_temp.append(Di_0)
        
        Z_terminal_temp = []
        Z_terminal_temp.append(torch.squeeze(self.sig(t1,W1)@(self.dividend_gradient(t1,W1)).unsqueeze(-1)))
        
        
        for i in range(I):
            Y_terminal_temp.append(E_0[i])
            Z_terminal_temp.append(torch.squeeze(self.sig(t1,W1)@(self.endowment_gradient(t1,W1,i)).unsqueeze(-1)))
       
        
        Y_terminal = torch.cat(Y_terminal_temp, 1)
        Y_terminal = Y_terminal.detach()
        Z_terminal = torch.cat(Z_terminal_temp, 1)
        Z_terminal = Z_terminal.detach()

        loss += nn.MSELoss(reduction='mean')(Y1, Y_terminal)
       
        return loss
    

    def train(self,NIter,epoch,patience):
        """
        Train the model for a specified number of epochs.

        Args:
            NIter (int): Number of iterations per epoch.
            epoch (int): Number of epochs.
            learning_rate (float): Learning rate for the Adam optimizer.
            patience (int): Number of epochs to wait for improvement in validation loss before early stopping.

        Notes:
            - Training is done in batches of size M.
            - Validation loss is computed after each epoch.
            - Early stopping and model checkpointing are used to save the best model.
        """
        start_time = time.perf_counter()
        
        # Variables to track early stopping
        early_stopping_counter = 0
        
        for j in range(epoch):
            print(f"Epoch {j+1}\n-------------------------------")
            for i in range(NIter):
                t_batch, W_batch = BrownianMotionGenerator.generate(self.M,self.N,self.D,self.T)
        
                loss=self.loss_func(t_batch,W_batch)
                self.optimizer.zero_grad()
                loss.backward()
        
                self.optimizer.step()
                self.history['train_loss'].append(loss.item())

                # # ===== train error =====
                # with torch.no_grad():
                #     Y_path, Y_real, Z_path, Z_real = self.calculate_path(t_batch, W_batch)
                    
                #     train_err = self.compute_total_error(Y_path, Y_real, Z_path, Z_real).item()

                # self.history['train_error'].append(train_err)


                # Print loss
                if (i+1) % 200 == 0:
                    elapsed=time.perf_counter() - start_time
                    print(f"Loss {(i+1)}: {loss.item()}, time: {elapsed}")
                    
            # ===== validation =====
            val_loss = self.loss_func(self.t_valid, self.W_valid).item()

            # with torch.no_grad():
            #     Y_path, Y_real, Z_path, Z_real = self.calculate_path(self.t_valid, self.W_valid)
                
            #     val_err = self.compute_total_error(Y_path, Y_real, Z_path, Z_real).item()

    

            print(f"Validation Loss {(j+1)}: {val_loss:.4f}")
            self.history['valid_loss'].append(val_loss)
            # self.history['valid_error'].append(val_err)

            # Early stopping and model checkpointing
            if val_loss < self.history['best_val_loss']:
                print(f"val_loss: {val_loss:.4f}, updating best_val_loss from: {self.history['best_val_loss']:.4f}.")
                self.history['best_val_loss'] = val_loss
                self.history['best_epoch'] = j
                torch.save({
                    "model_z": self.model_z.state_dict(),
                    "model_y0": self.model_y0.state_dict(),
                }, self.checkpoint_path)
                early_stopping_counter = 0
            else:
                print(f"val_loss: {val_loss:.4f}, worse than best_val_loss: {self.history['best_val_loss']:.4f}.")
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {j+1}")
                    break
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_step = total_time / j

        self.history['train_time'] = total_time
        self.history['time_per_epoch'] = avg_time_per_step

        

        ckpt = torch.load(self.checkpoint_path, map_location = self.device)
        # self.model.load_state_dict(torch.load(PATH)) ###reload the best model
        self.model_z.load_state_dict(ckpt["model_z"])
        self.model_y0.load_state_dict(ckpt["model_y0"])
    



    def predict(self, t, W):
        """
        Predict the price paths Y(t,W) of the Radner equilibrium FBSDE by solving the FBSDE using the learned neural networks.

        Returns:
            Y_path (list of torch.Tensor): predicted price paths with shape (I+1, M, N)
            Y_real (list of torch.Tensor): exact price paths with shape (I+1, M, N)
        """
        I=self.I
        
        t_star, W_star = t, W
        
        t0 = t_star[:,0,:].float() #K*1 initial time
        W0 = W_star[:,0,:].float() #K*D
        
        
        Y_path = [[] for _ in range(I+1)]
        
        Theta_path = [[] for _ in range(I)]  # only agents
       
        
        
        
        Z0 = self.net_z(t0, W0)
        Y0 = self.net_y(t0, W0)
        
       
        
        for i in range(I+1):
            Y_path[i].append(Y0[:,i].unsqueeze(1))
            
       
        
        theta_0 = self.theta(Y0, Z0, self.K)
        for i in range(I):
            Theta_path[i].append(theta_0[i])
        
        
        
        for time in range(1, self.N + 1): #iterate from time 1 to time N
            t1=t_star[:,time,:].float() #M*1
            W1=W_star[:,time,:].float() #M*1
            Z1 = self.net_z(t1,W1)

            Y1_temp=[]
            Y0_1_pred = Y0[:,0].unsqueeze(1) + self.phi0(t0,W0,Z0,self.epsilon)*(t1-t0)+torch.sum(Z0[:,:self.D]*(W1-W0),1,keepdims=True)
            Y1_temp.append(Y0_1_pred)
            for j in range(1, I + 1):
                pred=Y0[:,j].unsqueeze(1)+self.phi(t0,W0,Z0,j,self.epsilon)*(t1-t0)+torch.sum(Z0[:,j*self.D:(j+1)*self.D]*(W1-W0),1,keepdims=True)
                Y1_temp.append(pred)
            
        
            Y1=torch.cat(Y1_temp,1)  #M*(I+2)
            
        
                
            for j in range(I+1):
                Y_path[j].append(Y1[:,j].unsqueeze(1))
            
            
            
            # theta at time t
            theta_t = self.theta(Y1, Z1, self.K)
            for i in range(I):
                Theta_path[i].append(theta_t[i])
               
            
            Y0, Z0, t0, W0 = Y1, Z1, t1, W1
               
        
        for i in range(I+1):
            Y_path[i]=torch.stack(Y_path[i],dim=1)
        
        
        
        for i in range(I):
            Theta_path[i] = torch.stack(Theta_path[i], dim=1)
            
        
        
        return Y_path, Theta_path
    


   