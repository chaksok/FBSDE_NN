from tracemalloc import start
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
import torch.nn.functional as F


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
        self.N_option = config['N_option']
        self.a_expo = config['a_expo']


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
    
    
    def endowment_process(self, t, X, i):

        

        N_option = self.N_option
        a = self.a_expo

        temp = self.sigE[i,:] * X
        # return torch.sum(temp, -1, keepdim = True)

        X2 = temp[:,1:2]   # second Brownian source
    
        # put_payoff = F.softplus(-X2, beta=50.0)
        payoff = (1- a) * X2 + a * torch.minimum(X2, torch.zeros_like(X2))
        # put_payoff = X2

        if i == 0:

            return -N_option * payoff

        else:

            return N_option * payoff
    

    # def endowment_process(self, t, X, i,):
    
    #     N_option = self.N_option
    #     temp = self.sigE[i, :] * X
    #     X2 = temp[:, 1:2]
    #     a = self.a_expo
        
    #     # 线性部分
    #     # linear_part = -X2
        
    #     # 非线性部分（put payoff）
    #     nonlinear_part = -torch.minimum(X2, torch.zeros_like(X2))
        
    #     # 插值
    #     # exposure = (1 - a) * linear_part + a * nonlinear_part
    #     exposure = nonlinear_part
        
    #     if i == 0:
    #         return -N_option * exposure
    #     else:
    #         return N_option * exposure
    
    # def endowment_process(self, t, X, i):
    #     X2 = (self.sigE[i,:] * X)[:, 1:2]
    #     expo = torch.abs(X2)
    #     if i == 0:
    #         return self.N_option * expo
    #     else:
    #         return -self.N_option * expo
        
        
    
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
        
        norm2 = torch.sum(Z0**2, dim=1, keepdim=True)
        norm = torch.sqrt(norm2 + 1e-8)
        mask = (norm > 1e-6).float()

        unit = Z0 / norm
        
        
        
        part2=0.5*torch.sum(Z[:,i*D:(i+1)*D]**2,-1,keepdims=True)
        
        
        temp = Z0.clone()
        for j in range(I):
            temp=temp+alpha[j]*Z[:,(j+1)*D:(j+2)*D]
        part1=torch.sum((temp-Z[:,i*D:(i+1)*D])*unit,-1,keepdims=True)
        return part2-0.5*part1**2 * mask  ##M*1
    

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
        
        Z0 = Z[:,0:D]
        temp = Z0.clone()
        for i in range(I):
            temp = temp+alpha[i]*Z[:,(i+1)*D:(i+2)*D]
        
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
     
    
    def theta(self, Y, Z, K):

        I = self.I
        D = self.D

        Z0 = Z[:, 0:D]                          # \zeta
        norm2 = torch.sum(Z0**2, dim=1, keepdim=True)
        
        # Smooth division to avoid spikes near zero without hard clamping
        direction = Z0 / (norm2 + 1e-8)

        mask = (norm2 > 1e-8).float()

        
        direction = direction * mask

        tmp = 0
        for k in range(I):
            gamma_k = Z[:, (k+1)*D:(k+2)*D]
            tmp = tmp + self.alpha[k] * gamma_k

        theta = []

        for i in range(I):
            gamma_i = Z[:, (i+1)*D:(i+2)*D]

            theta_i = 1 + torch.sum(
                (tmp - gamma_i) * direction,
                dim=1,
                keepdim=True
            )

            theta.append(theta_i)

        return theta
    



class RadnerEquilibriumSolver(TerminalSolver, ABC):
    def __init__(self, config):
        super().__init__(config)
        self.layers = config['layers']   
      

        # Neural network and optimizer
        self.model = myNN(self.layers).to(self.device)


        super().__init__(config)    
        self.layers = config['layers']

        # Neural network and optimizer
        self.model = myNN(self.layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])


    @abstractmethod
    def net_u(self, t, x): 
        """
        Compute the neural network output u(t, x) and its gradient with respect to x.

        Args:
            t (torch.Tensor): Time of shape (M, 1)
            x (torch.Tensor): State of shape (M, D)

        Returns:
            y (torch.Tensor): Neural network output of shape (M, I)
            z (torch.Tensor): Gradient of neural network output with respect to x of shape (M, I*D)
        """
        pass
    

    def loss_func(self, t, W): #Xi is the initial X at time 0
        """
        Compute the loss function for Radner equilibrium solver.

        The loss function consists of mean squared error between the predicted next state and the actual next state, as well as mean squared error between the predicted terminal state and the actual terminal state.

        Args:
            t (torch.Tensor): Time of shape (M, 1)
            W (torch.Tensor): Dividend process of shape (M, D)

        Returns:
            loss (torch.Tensor): Mean squared error loss
        """
        I=self.I
        loss=torch.tensor([0.0],device=self.device)

        
        t0=t[:,0,:].float() #M*1 initial time
        W0=W[:,0,:].float() #M*D

        
        Y0, Z0=self.net_u(t0,W0)  #M*(I*1), M*(I*D)
        for i in range(I+1):
            Z0[:,i*self.D:(i+1)*self.D]=(Z0[:,i*self.D:(i+1)*self.D].unsqueeze(1)@self.sig(t0,W0)).squeeze()

        
        for time in range(1,self.N+1): #iterate from time 1 to time N
            t1=t[:,time,:].float() #M*1
            W1=W[:,time,:].float() #M*1
            
            
            Y0_0=Y0[:,0].unsqueeze(1)
            Z0_0=Z0[:,0:self.D]
            
            Y1_temp=[]
            Y0_1_pred=Y0_0+self.phi0(t0,W0,Z0,self.epsilon)*(t1-t0)+torch.sum(Z0_0*(W1-W0),1,keepdims=True)
            Y1_temp.append(Y0_1_pred)
            for j in range(1,I+1):
                pred=Y0[:,j].unsqueeze(1)+self.phi(t0,W0,Z0,j,self.epsilon)*(t1-t0)+torch.sum(Z0[:,j*self.D:(j+1)*self.D]*(W1-W0),1,keepdims=True)
                Y1_temp.append(pred)
            
        
            Y1_pred=torch.cat(Y1_temp,1)  #M*(I+2)
            
            Y1,Z1=self.net_u(t1,W1)
            for i in range(I+1):
                Z1[:,i*self.D:(i+1)*self.D]=(Z1[:,i*self.D:(i+1)*self.D].unsqueeze(1)@self.sig(t1,W1)).squeeze()
        
            loss+=nn.MSELoss(reduction='mean')(Y1,Y1_pred)
            
            t0, W0, Y0, Z0 = t1, W1, Y1, Z1
                   
        Di_0=self.dividend_process(t1,W1) #M*1
        E_0=[self.endowment_process(t1,W1,i) for i in range(I)]  #list of M*1
        
        Y_terminal_temp=[]
        Y_terminal_temp.append(Di_0)
        
        Z_terminal_temp=[]
        Z_terminal_temp.append(torch.squeeze(self.sig(t1,W1)@(self.dividend_gradient(t1,W1)).unsqueeze(-1)))
        
        
        for i in range(I):
            Y_terminal_temp.append(E_0[i])
            Z_terminal_temp.append(torch.squeeze(self.sig(t1,W1)@(self.endowment_gradient(t1,W1,i)).unsqueeze(-1)))
       
        
        Y_terminal=torch.cat(Y_terminal_temp,1)
        Y_terminal=Y_terminal.detach()
        Z_terminal=torch.cat(Z_terminal_temp,1)
        Z_terminal=Z_terminal.detach()

        loss += nn.MSELoss(reduction='mean')(Y1,Y_terminal)
       
        return loss
    

    def train(self, NIter, epoch, patience, min_delta=1e-6):

        start_time = time.perf_counter()

        early_stopping_counter = 0
        best_val_loss = float("inf")
        best_epoch = 0

        stop_training = False

        for j in range(epoch):

            print(f"\nEpoch {j+1}\n-------------------------------")

            # ===== Training loop =====
            for i in range(NIter):

                start = i * self.M
                end = (i + 1) * self.M

                t_batch, W_batch = self.t_train[start:end], self.W_train[start:end]

                # t_batch, W_batch = BrownianMotionGenerator.generate(
                #     self.M, self.N, self.D, self.T
                # )

                loss = self.loss_func(t_batch, W_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.history['train_loss'].append(loss.item())


                # Print progress
                if (i+1) % 200 == 0:

                    elapsed = time.perf_counter() - start_time

                    print(
                        f"Iter {i+1}/{NIter}, "
                        f"Loss: {loss.item():.3e}, "
                        f"Time: {elapsed:.1f}s"
                    )

            # ===== Validation =====

            val_loss = self.loss_func(self.t_valid, self.W_valid).item()

            
            self.history['valid_loss'].append(val_loss)
            

            print(
                f"Validation Loss: {val_loss:.3e}, "

            )

            # ===== Professional early stopping logic (checkpoint version) =====

            improvement = best_val_loss - val_loss

            if improvement > min_delta:

                print(
                    f"Validation Loss improved "
                    f"({best_val_loss:.3e} → {val_loss:.3e}), saving model."
                )

                best_val_loss = val_loss
                best_epoch = j
                early_stopping_counter = 0

                # keep your original checkpoint method
                torch.save(self.model.state_dict(), self.checkpoint_path)

            else:

                early_stopping_counter += 1

                print(
                    f"No improvement for {early_stopping_counter}/{patience} epochs"
                )

                # if early_stopping_counter >= patience:

                #     print(f"\nEarly stopping triggered at epoch {j+1}")

                #     stop_training = True
                #     break

            # if stop_training:
            #     break

        # ===== Reload best model =====

        print(f"\nReloading best model from checkpoint...")
        self.model.load_state_dict(torch.load(self.checkpoint_path))

        print(
            f"Best validation loss: {best_val_loss:.3e} "
            f"(epoch {best_epoch+1})"
        )

        # ===== Training statistics =====

        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time_per_epoch = total_time / (j+1)

        self.history['train_time'] = total_time
        self.history['time_per_epoch'] = avg_time_per_epoch
        self.history['best_val_loss'] = best_val_loss
        self.history['best_epoch'] = best_epoch

        print(
            f"\nTraining completed in {total_time:.1f}s "
            f"({avg_time_per_epoch:.1f}s per epoch)"
        )

    def predict(self, t, W):
        """
        Predict the terminal state given the dividend process W using the trained model.
        
        Parameters:
            None
        
        Returns:
            Y_path (list of torch.Tensor): Predicted terminal state
            Y_real (list of torch.Tensor): Real terminal state
        """
        I = self.I
        
        t_star, W_star = t, W
        
        t0 = t_star[:,0,:].float() #K*1 initial time
        W0 = W_star[:,0,:].float() #K*D
        
        
        Y_path = [[] for _ in range(I+1)]
       
        Theta_path = [[] for _ in range(I)]  # only agents
       
        Y0, Z0 = self.net_u(t0, W0)
        
        
        for i in range(I + 1):
            Y_path[i].append(Y0[:,i].unsqueeze(1))
            
        
        theta_0 = self.theta(Y0, Z0, self.K)
        for i in range(I):
            Theta_path[i].append(theta_0[i])
        
        
        
        for time in range(1, self.N + 1): #iterate from time 1 to time N
            t1=t_star[:,time,:].float() #M*1
            W1=W_star[:,time,:].float() #M*1
            
            Y1, Z1 = self.net_u(t1, W1)
            
                
            for j in range(I + 1):
                Y_path[j].append(Y1[:,j].unsqueeze(1))
            
            # theta at time t
            theta_t = self.theta(Y1, Z1, self.K)
            for i in range(I):
                Theta_path[i].append(theta_t[i])

            Y0, t0, W0 = Y1, t1, W1
        
        for i in range(I + 1):
            Y_path[i] = torch.stack(Y_path[i],dim=1)
        
        
        for i in range(I):
            Theta_path[i] = torch.stack(Theta_path[i], dim=1)
        

        Di_0 = self.dividend_process(t1,W1) #M*1
        E_0 = [self.endowment_process(t1,W1,i) for i in range(I)]  #list of M*1
        
        Y_terminal_temp = []
        Y_terminal_temp.append(Di_0)
        
        
        
        for i in range(I):
            Y_terminal_temp.append(E_0[i])
            
        
        Y_terminal = torch.cat(Y_terminal_temp, 1)
        Y_terminal = Y_terminal.detach()

        loss = nn.MSELoss(reduction='mean')(Y1, Y_terminal)
        print(f"Terminal loss: {loss.item():.4f}")
            
        return Y_path, Theta_path


class RadnerEquilibriumSolver1(RadnerEquilibriumSolver): 
    def net_u(self,t,x): 
        """
        Compute the neural network output u(t, x) and its gradient with respect to x.
        
        Args:
            t (torch.Tensor): Time of shape (M, 1)
            x (torch.Tensor): State of shape (M, D)
        
        Returns:
            y (torch.Tensor): Neural network output of shape (M, I)
            z (torch.Tensor): Gradient of neural network output with respect to x of shape (M, I*D)
        """
        
        Ii=self.I + 1
        x.requires_grad = True
        result=self.model(t,x)
        y = result[:,0 : Ii]   #M*(I*1)
        z = result[:, Ii : Ii * (self.D + 1)] #M*(I*D)
        return y, z
        

class RadnerEquilibriumSolver2(RadnerEquilibriumSolver):
    def net_u(self, t, x):
        """
        Compute the neural network output u(t, x) and its Jacobian with respect to x.
        
        Args:
            t (torch.Tensor): Time of shape (M, 1)
            x (torch.Tensor): State of shape (M, D)
        
        Returns:
            y (torch.Tensor): Neural network output of shape (M, I)
            z (torch.Tensor): Jacobian of neural network output with respect to x of shape (M, I*D)
        """
        Ii = self.I + 1
        # Recommendation: set batch-dependent layers to eval() when computing Jacobian, and switch back to train() during training
        # self.model.eval()

        # Single-sample function: input (D,), (t_dim,) ; output (Ii,)
        def f(x_single, t_single):
            out = self.model(t_single.unsqueeze(0), x_single.unsqueeze(0))  # (1, Ii)
            return out.squeeze(0)                                           # (Ii,)

        # Compute Jacobian J: (B, Ii, D), note the parameter order (x, t)
        J = vmap(jacrev(f, argnums=0), in_dims=(0, 0))(x, t)

        # Forward pass for batch
        y = self.model(t, x)                                 # (B, Ii)

        # Flatten to the desired shape
        z = J.reshape(x.size(0), Ii * self.D)                # (B, Ii*D)
        return y, z


        
class RadnerEquilibriumSolver4(TerminalSolver):
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
        z = self.model_z(t, x)
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


    def loss_func(self, t, W):

        I = self.I
        D = self.D

        loss = torch.tensor([0.0], device=self.device)

        # -------- separate trackers --------
        phi0_max_tracker = []
        phi0_min_tracker = []

        phi_i_max_tracker = [[] for _ in range(I)]
        phi_i_min_tracker = [[] for _ in range(I)]

        t0 = t[:, 0, :].float()
        W0 = W[:, 0, :].float()

        Y0 = self.net_y(t0, W0)
        Z0 = self.net_z(t0, W0)

        for i in range(I+1):
            Z0[:, i*D:(i+1)*D] = (
                Z0[:, i*D:(i+1)*D].unsqueeze(1)
                @ self.sig(t0, W0)
            ).squeeze()

        # =========================
        # time loop
        # =========================

        for time in range(1, self.N+1):

            # ---- phi0 ----
            phi0_val = self.phi0(t0, W0, Z0, self.epsilon)

            phi0_max_tracker.append(torch.max(phi0_val))
            phi0_min_tracker.append(torch.min(phi0_val))

            # ---- phi_i ----
            phi_vals_i = []

            for j in range(I):
                phi_j = self.phi(t0, W0, Z0, j+1, self.epsilon)

                phi_i_max_tracker[j].append(torch.max(phi_j))
                phi_i_min_tracker[j].append(torch.min(phi_j))

                phi_vals_i.append(phi_j)

            # ---- forward step ----
            t1 = t[:, time, :].float()
            W1 = W[:, time, :].float()

            Z0_0 = Z0[:, 0:D]

            Y1_temp = []

            Y0_1_pred = (
                Y0[:, 0].unsqueeze(1)
                + phi0_val * (t1 - t0)
                + torch.sum(Z0_0 * (W1 - W0), 1, keepdims=True)
            )

            Y1_temp.append(Y0_1_pred)

            for j in range(I):
                pred = (
                    Y0[:, j+1].unsqueeze(1)
                    + phi_vals_i[j] * (t1 - t0)
                    + torch.sum(
                        Z0[:, (j+1)*D:(j+2)*D] * (W1 - W0),
                        1,
                        keepdims=True
                    )
                )
                Y1_temp.append(pred)

            Y1 = torch.cat(Y1_temp, 1)

            Z1 = self.net_z(t1, W1)

            for i in range(I+1):
                Z1[:, i*D:(i+1)*D] = (
                    Z1[:, i*D:(i+1)*D].unsqueeze(1)
                    @ self.sig(t1, W1)
                ).squeeze()

            t0, W0, Y0, Z0 = t1, W1, Y1, Z1

        # =========================
        # terminal loss
        # =========================

        Di_0 = self.dividend_process(t1, W1)
        E_0 = [self.endowment_process(t1, W1, i) for i in range(I)]

        Y_terminal_temp = [Di_0] + E_0
        Y_terminal = torch.cat(Y_terminal_temp, 1).detach()

        loss += nn.MSELoss(reduction='mean')(Y1, Y_terminal)

        # =========================
        # store global max/min per process
        # =========================

        self._last_phi0_max = torch.max(torch.stack(phi0_max_tracker)).item()
        self._last_phi0_min = torch.min(torch.stack(phi0_min_tracker)).item()

        self._last_phi_i_max = [
            torch.max(torch.stack(phi_i_max_tracker[j])).item()
            for j in range(I)
        ]

        self._last_phi_i_min = [
            torch.min(torch.stack(phi_i_min_tracker[j])).item()
            for j in range(I)
        ]

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

                if "phi_monitor_step" not in self.history:
                    self.history["phi_monitor_step"] = []

                self.history["phi_monitor_step"].append({
                    "phi0_max": self._last_phi0_max,
                    "phi0_min": self._last_phi0_min,
                    "phi_i_max": self._last_phi_i_max,
                    "phi_i_min": self._last_phi_i_min
                })
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
            # else:
            #     print(f"val_loss: {val_loss:.4f}, worse than best_val_loss: {self.history['best_val_loss']:.4f}.")
            #     early_stopping_counter += 1
            #     if early_stopping_counter >= patience:
            #         print(f"Early stopping at epoch {j+1}")
            #         break
        
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
        

        Di_0 = self.dividend_process(t1,W1) #M*1
        E_0 = [self.endowment_process(t1,W1,i) for i in range(I)]  #list of M*1
        
        Y_terminal_temp = []
        Y_terminal_temp.append(Di_0)
        
        
        
        for i in range(I):
            Y_terminal_temp.append(E_0[i])
            
        
        Y_terminal = torch.cat(Y_terminal_temp, 1)
        Y_terminal = Y_terminal.detach()

        loss = nn.MSELoss(reduction='mean')(Y1, Y_terminal)
        print(f"Terminal loss: {loss.item():.4f}")
            
        
        
        return Y_path, Theta_path
    

class RadnerEquilibriumSolver_NoPhi1(RadnerEquilibriumSolver1):

    def phi(self, t, X, Z, i, epsilon):
        return torch.zeros(X.shape[0], 1, device=self.device)

    def phi0(self, t, X, Z, epsilon):
        return torch.zeros(X.shape[0], 1, device=self.device)


class RadnerEquilibriumSolver_NoPhi4(RadnerEquilibriumSolver4):

    def phi(self, t, X, Z, i, epsilon):
        return torch.zeros(X.shape[0], 1, device=self.device)

    def phi0(self, t, X, Z, epsilon):
        return torch.zeros(X.shape[0], 1, device=self.device)


