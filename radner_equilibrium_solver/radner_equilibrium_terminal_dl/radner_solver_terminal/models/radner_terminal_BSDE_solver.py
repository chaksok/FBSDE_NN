"""
radner_terminal_BSDE_solver.py

Module for solving Radner equilibria using Forward-Backward Stochastic Differential Equations (FBSDEs) with neural networks.

This module provides an abstract base class for FBSDE solvers and a concrete implementation for Radner equilibrium problems.
It uses PyTorch for neural network training and supports customizable model parameters.

Key Features:
- Abstract base class `FBSDEBase` for general FBSDE problems.
- Concrete class `RadnerEquilibriumFBSDE` implementing Radner equilibrium conditions.
- Model parameters passed as constructor arguments for flexibility.
- Support for early stopping, validation, and checkpointing.

Notes:
- Uses PyTorch; device is set to CPU by default (can be changed).
- Requires torch, numpy, matplotlib, seaborn, plotly.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import time
from abc import ABC,abstractmethod
from torch.func import jacrev, vmap

from core.network import FeedForwardNN as myNN
from core.data_generator import BrownianMotionGenerator
from core.base_solver import FBSDESolver


torch.manual_seed(1)
np.random.seed(1)

    
class RadnerEquilibriumSolver(FBSDESolver, ABC):
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
        self.layers = config['layers']

        # Neural network and optimizer
        self.model = myNN(self.layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])

        # Test data for prediction
        self.t_star,self.W_star = BrownianMotionGenerator.generate(self.K, self.N, self.D, self.T)
    
    
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

        
        Y0,Z0=self.net_u(t0,W0)  #M*(I*1), M*(I*D)
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
    

    def train(self, NIter, epoch, patience):
        """
        Train the model using DataLoader for batching, with batch size M.

        Parameters:
            NIter (int): number of iterations (steps)
            epoch (int): number of epochs
            patience (int): number of epochs to wait for improvement before stopping training

        Returns:
            None
        """
        start_time=time.time()
        
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
                if (i+1) % 200 == 0:
                    elapsed=time.time() - start_time
                    print(f"Loss {(i+1)}: {loss.item()}, time: {elapsed}")
                    
            val_loss = (self.loss_func(self.t_valid,self.W_valid)).item()
            print(f"Validation Loss {(j+1)}: {val_loss:.4f}")
            self.history['valid_error'].append(val_loss)
            # Early stopping and model checkpointing
            if val_loss < self.history['best_val_loss']:
                print(f"val_loss: {val_loss:.4f}, updating best_val_loss from: {self.history['best_val_loss']:.4f}.")
                self.history['best_val_loss'] = val_loss
                self.history['best_epoch'] = j
                torch.save(self.model.state_dict(), self.checkpoint_path)
                early_stopping_counter = 0
            else:
                print(f"val_loss: {val_loss:.4f}, worse than best_val_loss: {self.history['best_val_loss']:.4f}.")
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {j+1}")
                    break
        
        self.model.load_state_dict(torch.load(self.checkpoint_path, weights_only=True)) ###reload the best model
    

    def predict(self):
        """
        Predict the terminal state given the dividend process W using the trained model.
        
        Parameters:
            None
        
        Returns:
            Y_path (list of torch.Tensor): Predicted terminal state
            Y_real (list of torch.Tensor): Real terminal state
        """
        I = self.I
        
        t_star, W_star = self.t_star, self.W_star
        
        t0 = t_star[:,0,:].float() #K*1 initial time
        W0 = W_star[:,0,:].float() #K*D
        
        
        Y_path = [[] for _ in range(I+1)]
        Y_real = [[] for _ in range(I+1)]
        
        Y0, _ = self.net_u(t0, W0)
        S_0 = self.S_exact(t0, W0)
        Y_real[0].append(S_0)
        
        for i in range(I + 1):
            Y_path[i].append(Y0[:,i].unsqueeze(1))
            
        for i in range(1, I+1):
            Y_real[i].append(self.Y_exact(t0, W0, i-1).unsqueeze(1))
        
        for time in range(1, self.N + 1): #iterate from time 1 to time N
            t1=t_star[:,time,:].float() #M*1
            W1=W_star[:,time,:].float() #M*1
            
            Y1, _ = self.net_u(t1,W1)
            S_1 = self.S_exact(t1,W1)
                
            for j in range(I + 1):
                Y_path[j].append(Y1[:,j].unsqueeze(1))
            
            Y_real[0].append(S_1)
            
            for i in range(1, I + 1):
                Y_real[i].append(self.Y_exact(t1, W1, i-1).unsqueeze(1))
            
            Y0, t0, W0 = Y1, t1, W1
        
        for i in range(I + 1):
            Y_path[i] = torch.stack(Y_path[i],dim=1)
        
        for i in range(I + 1):
            Y_real[i] = torch.stack(Y_real[i],dim=1)
        
        return Y_path, Y_real
    


class RadnerNeuralNetwork1(RadnerEquilibriumSolver): 
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
        

class RadnerNeuralNetwork2(RadnerEquilibriumSolver):
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


class RadnerNeuralNetwork3(RadnerEquilibriumSolver):
    def bsde_2p8_terms_ultrafast(self, t, x, dW, h): 
        """
        Compute the 2nd order terms of the Forward-Backward SDE using ultra-fast algorithm.

        Parameters:
            t (torch.Tensor): Time of shape (B, 1)
            x (torch.Tensor): State of shape (B, D)
            dW (torch.Tensor): Brownian motion increment of shape (B, D)
            h (torch.Tensor or float): Time step

        Returns:
            out (torch.Tensor): 2nd order terms of shape (B, K)
        """
        B, D = x.shape
        device, dtype = x.device, x.dtype
        
        x = x.clone().detach().requires_grad_(True)
        U = self.model(t, x)
        K = U.shape[1]
        mu = self.mu(t, x)
        Sig = self.sig(t, x)
        v = torch.einsum('bdd,bd->bd', Sig, dW)
        
        if not torch.is_tensor(h):
            h = torch.tensor(h, device=device, dtype=dtype)
        if h.dim() == 0:
            h = h.expand(B, 1)
        
        out = torch.zeros(B, K, device=device, dtype=dtype)
        
        for k in range(K):
            gk = torch.autograd.grad(
                outputs=U[:, k].sum(), inputs=x,
                retain_graph=True, create_graph=True, only_inputs=True
            )[0]
            
            grad_v_product = (gk * v).sum()
            Hv = torch.autograd.grad(
                outputs=grad_v_product, inputs=x,
                retain_graph=True, create_graph=True, only_inputs=True
            )[0]
            
            term1 = -h * (mu * Hv).sum(dim=1, keepdim=True)
            term2 = -0.5 * (Hv * v).sum(dim=1, keepdim=True)
            # Skip term3 (trace term)
            
            out[:, k] = (term1 + term2).squeeze(1)
        
        return out

    # === Integrated version: modify net_u to be compatible ===
    def net_u(self, t, x):
        """
        Compute the neural network output u(t, x) and its Jacobian with respect to x.

        Args:
            t (torch.Tensor): Time of shape (M, 1)
            x (torch.Tensor): State of shape (M, D)

        Returns:
            y (torch.Tensor): Neural network output of shape (M, Ii)
            z (torch.Tensor): Jacobian of neural network output with respect to x of shape (M, Ii*D)
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
    

    def loss_func(self,t,W,M): #Xi is the initial X at time 0
        """
        Compute the loss using Forward-Backward SDE structure.

        Parameters:
        t (torch.Tensor): Time of shape (M, N+1)
        W (torch.Tensor): Brownian Motion of shape (M, N+1, D)
        M (int): Number of trajectories

        Returns:
        loss (torch.Tensor): Computed loss
        """
        I=self.I
        loss=torch.tensor([0.0], device = self.device)

        t0=t[:,0,:].float() #M*1 initial time
        W0=W[:,0,:].float() #M*D

        Y0, Z0 =self.net_u(t0,W0)  #M*(I*1), M*(I*D)
        for i in range(I+1):
            Z0[:,i* self.D:(i+1)* self.D]=(Z0[:,i* self.D:(i+1)* self.D].unsqueeze(1) @ self.sig(t0,W0)).squeeze()

        
        for time in range(1,self.N+1): #iterate from time 1 to time N
            t1=t[:,time,:].float() #M*1
            W1=W[:,time,:].float() #M*1
            
            expr = self.bsde_2p8_terms_ultrafast(t0, W0, W1 - W0, t1 - t0)
            
            Y0_0=Y0[:,0].unsqueeze(1)
            Z0_0=Z0[:,0: self.D] 
            
            Y1_temp=[]
            Y0_1_pred=Y0_0+self.phi0(t0,W0,Z0,self.epsilon)*(t1-t0)+torch.sum(Z0_0*(W1-W0),1,keepdims=True) + expr[:, 0].unsqueeze(1)
            Y1_temp.append(Y0_1_pred)
            for j in range(1,I+1):
                pred=Y0[:,j].unsqueeze(1)+self.phi(t0,W0,Z0,j,self.epsilon)*(t1-t0)+torch.sum(Z0[:,j*self.D:(j+1)*self.D]*(W1-W0),1,keepdims=True) + expr[:, i].unsqueeze(1)
                Y1_temp.append(pred)
            
            
        
            Y1_pred=torch.cat(Y1_temp,1)  #M*(I+2)
            
            Y1,Z1=self.net_u(t1,W1)
            for i in range(I+1):
                Z1[:,i*self.D:(i+1)*self.D]=(Z1[:,i*self.D:(i+1) * self.D].unsqueeze(1)@self.sig(t1,W1)).squeeze()
        
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



class RadnerNeuralNetwork4(FBSDESolver):
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
    
   
    def train(self,NIter,epoch,learning_rate,patience):
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
        start_time = time.time()
        
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
                if (i+1) % 200 == 0:
                    elapsed=time.time() - start_time
                    print(f"Loss {(i+1)}: {loss.item()}, time: {elapsed}")
                    
            val_loss = (self.loss_func(self.t_valid,self.W_valid)).item()
            print(f"Validation Loss {(j+1)}: {val_loss:.4f}")
            self.history['valid_error'].append(val_loss)
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
        
        ckpt = torch.load(self.checkpoint_path, map_location = self.device)
        # self.model.load_state_dict(torch.load(PATH)) ###reload the best model
        self.model_z.load_state_dict(ckpt["model_z"])
        self.model_y0.load_state_dict(ckpt["model_y0"])
        
    
    def predict(self):
        """
        Predict the price paths Y(t,W) of the Radner equilibrium FBSDE by solving the FBSDE using the learned neural networks.

        Returns:
            Y_path (list of torch.Tensor): predicted price paths with shape (I+1, M, N)
            Y_real (list of torch.Tensor): exact price paths with shape (I+1, M, N)
        """
        I=self.I
        
        t_star, W_star = self.t_star, self.W_star
        
        t0 = t_star[:,0,:].float() #K*1 initial time
        W0 = W_star[:,0,:].float() #K*D
        
        
        Y_path = [[] for _ in range(I+1)]
        Y_real = [[] for _ in range(I+1)]
        
        
        Z0 = self.net_z(t0, W0)
        Y0 = self.net_y(t0, W0)
        S_0 = self.S_exact(t0, W0) 
        Y_real[0].append(S_0)
        
        for i in range(I+1):
            Y_path[i].append(Y0[:,i].unsqueeze(1))
            
        for i in range(1,I+1):
            Y_real[i].append(self.Y_exact(t0, W0, i-1).unsqueeze(1))
        
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
            
            S_1=self.S_exact(t1,W1)
                
            for j in range(I+1):
                Y_path[j].append(Y1[:,j].unsqueeze(1))
            
            Y_real[0].append(S_1)
            
            for i in range(1,I+1):
                Y_real[i].append(self.Y_exact(t1,W1,i-1).unsqueeze(1))
            
            Y0, Z0, t0, W0 = Y1, Z1, t1, W1
               
        
        for i in range(I+1):
            Y_path[i]=torch.stack(Y_path[i],dim=1)
        
        for i in range(I+1):
            Y_real[i]=torch.stack(Y_real[i],dim=1)
        
        
        return Y_path, Y_real

    