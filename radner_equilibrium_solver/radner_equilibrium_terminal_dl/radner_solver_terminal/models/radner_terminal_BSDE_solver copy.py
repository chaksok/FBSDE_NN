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
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import plotly.express as px
sns.set()
# plt.style.use('ggplot')

torch.manual_seed(1)
np.random.seed(1)

PATH = "tryexact.pt"
device = 'cpu'

# =====================================
# Neural Network Architecture
# =====================================
class myNN(nn.Module):
    """A simple feedforward neural network with GELU activations."""
    def __init__(self, layers):
        super().__init__()
        self.linear_gelu_stack = nn.Sequential()
        for i in range(len(layers) - 2):
            self.linear_gelu_stack.add_module(f"linear{i}", nn.Linear(layers[i], layers[i + 1]))
            self.linear_gelu_stack.add_module(f'activation{i}', nn.GELU())
        self.linear_gelu_stack.add_module("flinear", nn.Linear(layers[len(layers) - 2], layers[len(layers) - 1]))

    def forward(self, t, x):
        """Forward pass through the network."""
        results = self.linear_gelu_stack(torch.cat((t, x), 1))
        return results
        results = self.linear_gelu_stack(torch.cat((t,x),1))
        return results
    

# =====================================
# Abstract Base Class for FBSDE Solvers
# =====================================
class FBSDNN(ABC):
    """Abstract base class for Forward-Backward Stochastic Differential Equation (FBSDE) solvers.

    This class defines the common structure and abstract methods for FBSDE problems.
    Subclasses must implement the abstract methods to define specific problem dynamics.
    """
    def __init__(self, config):

        self.T = config['T']
        self.M = config['M']
        self.N = config['N']
        self.D = config['D']
        self.I = config['I']
        self.K = config['K']
        self.layers = config['layers']
        self.device = config.get('device', 'cpu')

         # Neural network and optimizer
        self.model = myNN(self.layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])

        # Validation data
        self.t_valid,self.W_valid = self.gen_data(self.K,self.N,self.D,self.T)

        
        self.history = {
            'train_loss': [],
            'valid_error': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }

    
    def gen_data(self,M,N,D,T):
        Dt=np.zeros((M,N+1,1))
        DW=np.zeros((M,N+1,D))
    
        dt=T/N
        Dt[:,1:,:]=dt
        DW[:,1:,:]=np.sqrt(dt)*np.random.normal(size=(M,N,D))
    
        t=np.cumsum(Dt,axis=1)
        W=np.cumsum(DW,axis=1)

        t=torch.from_numpy(t)
        W=torch.from_numpy(W)
    
        return t.float().to(device), W.float().to(device)
    
    def net_u(self,t,x): 
        Ii=self.I+1
        x.requires_grad = True
        result=self.model(t,x)
        y = result[:,0:Ii]   #M*(I*1)
        z = result[:,Ii:Ii*(self.D+1)] #M*(I*D)
        return y,z
    
    
    
    @abstractmethod
    def g_Di(self, t, X):
        """Dividend process Di = g(t, X)."""
        pass  # M*1
    
    
    @abstractmethod
    def g_E(self,t,X,i): #endowment process for agent i 
        pass  
    

    
    @abstractmethod
    def phi0(self, t, X, Y, Z, Di, epsilon):
        """Drift for the price process S."""
        pass 
    
    @abstractmethod
    def phi(self,t,X,Y,Z,E,i,epsilon): ##M*1,M*D,M*(I*1), M*(I*D)    #drift for Y
        pass
    
   
    
    @abstractmethod
    def mu(self,t,X): ##M*1,M*D,M*(I*1), M*(I*D)    #drift for X
        return torch.zeros(X.shape).to(device)
    
    @abstractmethod
    def sig(self,t,X):  ##volatility for X 
        pass
    
    
    
    
    def loss_func(self,t,W,M): #Xi is the initial X at time 0
        pass
            
     
  
    def train(self, NIter, epoch, learning_rate, patience):
        """Train the model with early stopping."""
        start_time=time.time()
        

        # Variables to track early stopping
        early_stopping_counter = 0
        
        for j in range(epoch):
            print(f"Epoch {j+1}\n-------------------------------")
            for i in range(NIter):
                t_batch, W_batch=self.gen_data(self.M,self.N,self.D,self.T)
        
                loss=self.loss_func(t_batch,W_batch,self.M)
                self.optimizer.zero_grad()
                loss.backward()
        
                self.optimizer.step()
                self.history['train_loss'].append(loss.item())
                if (i+1)%200==0:
                    elapsed=time.time()-start_time
                    print(f"Loss {(i+1)}: {loss.item()}, time: {elapsed}")
                    
            val_loss=(self.loss_func(self.t_valid,self.W_valid,self.K)).item()
            print(f"Validation Loss {(j+1)}: {val_loss:.4f}")
            self.history['valid_error'].append(val_loss)
            # Early stopping and model checkpointing
            if val_loss < self.history['best_val_loss']:
                print(f"val_loss: {val_loss:.4f}, updating best_val_loss from: {self.history['best_val_loss']:.4f}.")
                self.history['best_val_loss'] = val_loss
                self.history['best_epoch'] = j
                torch.save(self.model.state_dict(),PATH)
                early_stopping_counter = 0
            else:
                print(f"val_loss: {val_loss:.4f}, worse than best_val_loss: {self.history['best_val_loss']:.4f}.")
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {j+1}")
                    break
        
        self.model.load_state_dict(torch.load(PATH, weights_only=True)) ###reload the best model
    
    @abstractmethod
    def theta(self,Y,Z,K):
        pass
    
    
class B_S_B2(FBSDNN):
    """Concrete implementation of FBSDE solver for Radner equilibrium problems."""
    def __init__(self, config, drift_D, sigD, muE, sigE, alpha, epsilon):
        super().__init__(config)
        self.drift_D = drift_D   ## coef of D=g(t,X) with respect to t
        self.sigD = sigD  ##coef of D=g(t,X) with respect to X
        self.muE = muE    ## coef of E=f_i(t,X) with respect to t
        self.sigE = sigE   ## coef of E=f_i(t,X) with respect to X
        self.alpha = alpha
        self.epsilon = epsilon
    
    def g_Di(self, t, X):  #dividend process Di=g(t,X)
        """Compute the dividend process."""
        mu = self.drift_D
        sig = self.sigD
        if len(sig[0, :]) != self.D:
            raise ValueError(f"the volatility of the asset should have {self.D} elements.\n")
        return torch.sum(sig * X, -1, keepdims=True)
    
    def Dg_Di(self,t,x):  ## first order derivative with respect to t and X
        x.requires_grad=True
        gDx=torch.autograd.grad(self.g_Di(t,x), x, grad_outputs=torch.ones_like(self.g_Di(t,x)))[0]
        x.requires_grad=False
        return gDx
    
    def g_E(self,t,X,i): #endowment process for agent i, E^i=g^i(t,X)
        mu=self.muE
        sig=self.sigE
        if len(mu)!=self.I:
            raise ValueError(f"the elements in the drift term of the endowments do not match the number of agents {self.I}.\n")
        if len(sig)!=self.I:
            raise ValueError(f"there should be {self.I} agents in the market.\n")
        if len(sig[0,:])!=self.D:
            raise ValueError(f"the volatility of endowments should be {self.D} dimension.\n")
        return torch.sum(sig[i,:]*X,-1,keepdims=True)
    
    def Dg_E(self,t,x,i): ## first order derivative with respect to t and X
        x.requires_grad=True
        gEx=torch.autograd.grad(self.g_E(t,x,i), x, grad_outputs=torch.ones_like(self.g_E(t,x,i)))[0]
        x.requires_grad=False
        return gEx
    
    def phi(self,t,X,Z,i,epsilon): #i is the order of the agent (start from 1), epsilon is the error we can bear in the computation 
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
        I=self.I
        alpha=self.alpha
        D=self.D
        
        Z0=Z[:,0:D]
        temp=Z0
        for i in range(I):
            temp=temp+alpha[i]*Z[:,(i+1)*D:(i+2)*D]
        
        return torch.sum(temp*Z0,-1,keepdims=True)  #M*1
        
    
    def mu(self,t,X):  ## drift of state process X
        return super().mu(t,X)
             
    def sig(self,t,X): #M*1,M*D,M*1, volatility of state process X
        size=X.shape
        a=torch.ones(size).to(device)
        return torch.diag_embed(a) #M*D*D
     
    
    def theta(self,Y,Z,K):  ## portfolio process
        I=self.I
        Z0=Z[:,0:self.D]
        
        theta=[]
        if torch.norm(Z0)<1e-7:
            temp=torch.randn(K,I).to(device)
            temp/=torch.sum(temp,-1,keepdims=True)
            for i in range(I):
                theta.append(temp[:,i].unsqueeze(-1))
            return theta
        else:
            temp=Z0/torch.sum(Z0**2,-1,keepdims=True) #K*D
            tmp=0
            for i in range(I):
                tmp=tmp+self.alpha[i]*Z[:,(i+1)*D:(i+2)*D]

            for i in range(I):
                theta_i=1+torch.sum((tmp-Z[:,(i+1)*self.D:(i+2)*self.D])*temp,-1,keepdims=True)
                theta.append(theta_i)
    
            return theta
    

    def loss_func(self,t,W,M): #Xi is the initial X at time 0
        I=self.I
        loss=torch.tensor([0.0],device=device)

        
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
            
            
            
            t0=t1
            W0=W1
            
            Y0=Y1
            Z0=Z1
            
            
        Di_0=self.g_Di(t1,W1) #M*1
        
        E_0=[]
        for i in range(I):
            E_0.append(self.g_E(t1,W1,i))
        
        
        Y_terminal_temp=[]
        Y_terminal_temp.append(Di_0)
        
        Z_terminal_temp=[]
        Z_terminal_temp.append(torch.squeeze(self.sig(t1,W1)@(self.Dg_Di(t1,W1)).unsqueeze(-1)))
        
        
        for i in range(I):
            Y_terminal_temp.append(E_0[i])
            Z_terminal_temp.append(torch.squeeze(self.sig(t1,W1)@(self.Dg_E(t1,W1,i)).unsqueeze(-1)))
       
        
        Y_terminal=torch.cat(Y_terminal_temp,1)
        Y_terminal=Y_terminal.detach()
        Z_terminal=torch.cat(Z_terminal_temp,1)
        Z_terminal=Z_terminal.detach()

        loss+= nn.MSELoss(reduction='mean')(Y1,Y_terminal)
        # loss+=0.05*nn.MSELoss(reduction='mean')(Z1,Z_terminal)
       
        
        return loss
    
if __name__ == "__main__":
    T=1 #terminal time
    M=100  #number of trajectoris
    N=100  #number of time step
    D=4
    I=3

    K=1000
    config = {
        'T': 1.0,
        'M': 100,
        'N': 100,
        'D': 4,
        'I': 3,
        'K': 1000,
        'layers': [D+1]+4*[256]+[(I+1)*(1+D)],  
        'learning_rate': 1e-4,
        'device': 'cpu',
    }

    drift_D=0.2
    sigD=torch.tensor([[0.3,0.,0.1,0.0]],device=device)
    muE=torch.tensor([0.1,0.1,0.1],device=device)
    sigE=torch.tensor([[0.3,0.3,0.,0.],[0.2,0.,0.3,0.],[0.1,0.,0.0,0.3]],device=device)
    alpha=[0.4,0.3,0.3]
    epsilon=1e-7
    V0=[1,1,1] #initial share of each agent, should satisfies alpha*V0=1
    epoch=50
    NIter=1000 #number of iteration
    patience=2
    
    myFBSDE=B_S_B2(config, drift_D, sigD, muE, sigE, alpha, epsilon)
    myFBSDE.train(NIter,5,1e-4,patience)     
