from abc import ABC, abstractmethod
import time
import torch
from torch import nn
import numpy as np
from .data_generator import BrownianMotionGenerator


class FBSDEBase(ABC):
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

        self.T = config['T']
        self.M = config['M']
        self.N = config['N']
        self.D = config['D']
        self.I = config['I']
        self.K = config['K']
        self.learning_rate = config['learning_rate']
        self.device = config.get('device', 'cpu')
        self.checkpoint_path = config.get('checkpoint_path', '/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/best_model.pt')

        
        # Validation data
        self.t_valid, self.W_valid = BrownianMotionGenerator.generate(
            self.K, self.N, self.D, self.T, self.device
        )

        self.history = {
            'train_loss': [],
            'valid_error': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }

    
    @abstractmethod
    def dividend_process(self, t, X):
        """
        Abstract method to compute the dividend process Di = g(t, X).

        Args:
            t (torch.Tensor): Time of shape (M, 1)
            X (torch.Tensor): State of shape (M, D)

        Returns:
            torch.Tensor: Dividend process of shape (M, 1)
        """
        pass  # M*1
    
    
    @abstractmethod
    def endowment_process(self,t,X,i): #endowment process for agent i 
        """Compute the endowment process E_i(t,X) for agent i.

        Args:
            t (torch.Tensor): Time of shape (M, 1)
            X (torch.Tensor): State of shape (M, D)
            i (int): Agent index

        Returns:
            torch.Tensor: Endowment process E_i of shape (M, 1)"""
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
        """
        Abstract method for computing the drift term of the state process X.

        Args:
            t (torch.Tensor): time
            X (torch.Tensor): state process X

        Returns:
            torch.Tensor: drift term of shape (M, D)
        """
        pass
    
    
    @abstractmethod
    def sig(self,t,X):  ##volatility for X 
        """
        Abstract method for computing the volatility matrix of the state process X.

        Parameters:
            t (torch.Tensor): time
            X (torch.Tensor): state process X

        Returns:
            torch.Tensor: volatility matrix of shape (M, D, D)
        """
        pass
    
    
    @abstractmethod
    def theta(self, Y, Z, K):
        """
        Abstract method for computing agent portfolios θ^i(t,W,S) in the Radner equilibrium.

        Parameters:
            Y (torch.Tensor): N x K matrix of price processes S(t,W)
            Z (torch.Tensor): N x K x D matrix of agent strategies θ^i(t,W,S)
            K (int): number of agents

        Returns:
            List of K tensors of shape (N, I), where I is the number of assets. Each tensor
            represents the portfolio of agent k at time t.
        """
        pass
    
    @abstractmethod
    def loss_func(self,t,W,M): #Xi is the initial X at time 0
        """
        Abstract method for computing the loss function of the FBSDE solver.

        Parameters:
            t (torch.Tensor): time
            W (torch.Tensor): Brownian motion
            M (int): batch size

        Returns:
            torch.Tensor: loss tensor
        """
        pass
            
     
    @abstractmethod
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
        pass

    @abstractmethod
    def predict(self):
        """
        Abstract method to predict the solution at given time and Brownian motion.

        Returns:
            torch.Tensor: predicted solution, real solution
        """
        pass
    
    

class FBSDESolver(FBSDEBase, ABC):
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
        if len(mu)!=self.I:
            raise ValueError(f"the elements in the drift term of the endowments do not match the number of agents {self.I}.\n")
        if len(sig)!=self.I:
            raise ValueError(f"there should be {self.I} agents in the market.\n")
        if len(sig[0,:])!=self.D:
            raise ValueError(f"the volatility of endowments should be {self.D} dimension.\n")
        return torch.sum(sig[i,:]*X,-1,keepdims=True)
    
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
    

    def S_exact(self,t,X): #K*1, K*D       
        """
        Compute the exact solution S(t,W) of the Radner equilibrium FBSDE.
        
        Parameters:
        t (float): time
        X (torch.tensor): Brownian motion with shape (M,N,D)
        
        Returns:
        torch.tensor: exact solution S(t,W) with shape (M,N,I)
        """
        I=self.I
        sigD=self.sigD
        temp=self.sigD
        alpha=self.alpha
        for i in range(I):
            temp=temp+alpha[i]*self.sigE[i,:]
        a=torch.sum(temp*sigD,-1,keepdims=True)
        return (t-1)*a+torch.sum(sigD*X, -1, keepdims=True) #K*1
    
    
    def Y_exact(self, t, X, i):
        """
        Compute the exact solution Y(t,W) of the Radner equilibrium FBSDE.

        Parameters:
        t (float): time
        X (torch.tensor): Brownian motion with shape (M,N,D)
        i (int): index of the component of Y

        Returns:
        torch.tensor: exact solution Y(t,W) with shape (M,N)
        """
        I = self.I
        alpha = self.alpha
        D = self.D
        sigD = self.sigD
        
        
        part2 = 0.5 * torch.sum(self.sigE[i,:]**2,-1,keepdims = True)
        
        unit = sigD/torch.sqrt(torch.sum(sigD**2,-1,keepdims = True))
        temp = sigD
        for j in range(I):
            temp = temp + alpha[j]*self.sigE[j,:]
        part1 = torch.sum((temp - self.sigE[i,:]) * unit, -1, keepdims = True)
        a = part2 - 0.5 * part1**2  ##M*1
        return (t-1) * a + torch.sum(self.sigE[i,:] * X, -1, keepdims = True)
    

    