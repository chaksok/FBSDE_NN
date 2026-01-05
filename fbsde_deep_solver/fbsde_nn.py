#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
import time
from abc import ABC,abstractmethod

# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


## This is a change

# In[3]:


class MyNN(nn.Module):
    def __init__(self, layers):
        """
        Initialize the neural network model.

        Parameters:
        layers (list): A list of integers where each value represents the number
                       of neurons in each layer of the network.
        """
        super(MyNN, self).__init__()

        # Sequential container to hold the layers
        self.linear_gelu_stack = nn.Sequential()

        # Add hidden layers with GELU activation
        for i in range(len(layers) - 2):
            # Linear layer from layers[i] to layers[i+1]
            self.linear_gelu_stack.add_module(f"linear{i}", nn.Linear(layers[i], layers[i + 1]))
            # GELU activation function after each linear layer
            self.linear_gelu_stack.add_module(f"gelu{i}", nn.GELU())

        # Add the final linear layer (output layer)
        self.linear_gelu_stack.add_module("output_linear", nn.Linear(layers[-2], layers[-1]))

    def forward(self, t, x):
        """
        Forward pass of the neural network.

        Parameters:
        t (Tensor): A tensor representing some time or other inputs (e.g. time-series data).
        x (Tensor): A tensor representing the main input features.

        Returns:
        Tensor: The output of the network after passing t and x through the layers.
        """
        # Concatenate t and x along dimension 1 (assuming t and x are row vectors or matrices)
        inputs = torch.cat((t, x), dim=1)

        # Pass the concatenated inputs through the sequential layer stack
        output = self.linear_gelu_stack(inputs)

        return output




# Define a custom dataset class
class FBSDataSet(Dataset):
    def __init__(self, t, W):
        self.t = t
        self.W = W

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        t_item = self.t[idx]
        W_item = self.W[idx]
        return t_item, W_item

# FBSDNN Model Class
class FBSDNN(ABC):
    def __init__(self, Xi, T, M, N, D, layers, learning_rate):
        """
        Initialize the FBSDNN model with neural network and optimizer.

        Parameters:
        Xi (float): Initial value of the state process.
        T (float): Final time.
        M (int): Number of Monte Carlo simulations (training samples).
        N (int): Number of time steps.
        D (int): Dimensionality of the state process.
        layers (list): Architecture of the neural network.
        learning_rate (float): Learning rate for the optimizer.
        """
        self.Xi = Xi
        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.layers = layers

        self.model = MyNN(self.layers).to(device)  # Neural network architecture
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def gen_data(self, M, N, D, T):
        """
        Generate Brownian motion paths and time grid.

        Returns:
        torch.Tensor: Time grid and Brownian motion paths.
        """
        Dt = np.zeros((M, N + 1, 1))
        DW = np.zeros((M, N + 1, D))

        dt = T / N
        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        t = np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)

        # Convert to tensors and move to device (GPU/CPU)
        t = torch.from_numpy(t).to(device)
        W = torch.from_numpy(W).to(device)

        return t, W

    def net_u(self, t, x):
        """
        Compute the neural network output u(t, x) and its gradient with respect to x.
        """
        x.requires_grad = True
        u = self.model(t, x)
        Du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True)[0]
        return u, Du

    @abstractmethod
    def u_exact(self,t,X): #K*1, K*D
        """
        real u
        """
        pass
    
    
    @abstractmethod
    def g_tc(self, x):
        """
        Terminal condition function g(X_T).
        """
        pass

    @abstractmethod
    def phi(self, t, X, Y, Z):
        """
        Compute the phi term in the differential equation.
        """
        pass

    @abstractmethod
    def sig(self, t, X, Y):
        """
        Compute the volatility matrix.
        """
        pass

    def loss_func(self, t, W, Xi):
        """
        Compute the loss using Forward-Backward SDE structure.
        """
        loss = torch.tensor([0.0], device=device)

        t0 = t[:, 0, :].float()
        W0 = W[:, 0, :].float()
        X0 = (torch.ones((self.M, 1), device=device) * Xi).float()

        Y0, Z0 = self.net_u(t0, X0)

        for time in range(1, self.N + 1):
            t1 = t[:, time, :].float()
            W1 = W[:, time, :].float()
            X1 = X0 + torch.squeeze(self.sig(t0, X0, Y0) @ (W1 - W0).unsqueeze(-1))

            X1 = X1.detach()  # Detach to stop gradients
            Y1_pred = Y0 + self.phi(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
                Z0 * torch.squeeze(self.sig(t0, X0, Y0) @ (W1 - W0).unsqueeze(-1)), dim=1, keepdims=True)
            Y1, Z1 = self.net_u(t1, X1)

            loss += nn.MSELoss(reduction='sum')(Y1, Y1_pred)

            t0, W0, X0, Y0, Z0 = t1, W1, X1, Y1, Z1

        # Add terminal condition loss
        loss += nn.MSELoss(reduction='sum')(Y1, self.g_tc(X1))

        return loss

    def train(self, NIter, epochs):
        """
        Train the model using DataLoader for batching, with batch size M.
        """
        
        start_time = time.time()
        

        t_total, W_total = self.gen_data(self.M * NIter, self.N, self.D, self.T)
        dataset = FBSDataSet(t_total, W_total)
        dataloader = DataLoader(dataset, batch_size=self.M, shuffle=True)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            for batch_idx, (t_batch, W_batch) in enumerate(dataloader):
                loss = self.loss_func(t_batch, W_batch, self.Xi)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Batch {batch_idx + 1}, Loss: {loss.item()}, Time: {elapsed:.2f}s")

    def predict(self, Xi_star, K):
        """
        Predict the Y path using trained model and DataLoader for batch processing, with batch size M.
        """
        t_star,W_star=self.gen_data(K,self.N,self.D,self.T)
        
        t0=t_star[:,0,:].float() #M*1 initial time
        W0=W_star[:,0,:].float() #M*D
        
        
        
        X0=(torch.ones((K,1),device=device)*Xi_star).float()  #K*D
        
        
        Y_star_path=[]
        Y_real=[]
        
        Y0=self.model(t0,X0)
        Y_r=self.u_exact(t0,X0)
        
        Y_real.append(Y_r)
        Y_star_path.append(Y0)
    
        for time in range(1,self.N+1):
            with torch.no_grad():
                t1=t_star[:,time,:].float() #M*1 initial time
                W1=W_star[:,time,:].float() #M*D
                X1=X0+torch.squeeze(self.sig(t0,X0,Y0)@(W1-W0).unsqueeze(-1))
                X1=X1.detach()
                Y1=self.model(t1,X1)
                y_r=self.u_exact(t1,X1)
                
                Y_star_path.append(Y1)
                Y_real.append(y_r)
                
                X0=X1
                Y0=Y1
                t0=t1
                W0=W1
                
            
        Y_real=torch.stack(Y_real,dim=1) 
        Y_star_path=torch.stack(Y_star_path,dim=1)

    
        return Y_star_path,Y_real


## Example 1
class toy_example1(FBSDNN):
    def __init__(self, Xi, T, M, N, D, layers, learning_rate, r, sigma):
        super().__init__(Xi, T, M, N, D, layers, learning_rate)
        self.r = r
        self.sigma = sigma
   
    def u_exact(self,t,X): #K*1, K*D
        """
        real u
        """
        return torch.exp((self.r +self.sigma**2)*(self.T-t))*self.g_tc(X)
    
   
    def g_tc(self, x):
        """
        Terminal condition function g(X_T).
        """
        return torch.sum(x**2,-1,keepdims=True)

   
    def phi(self, t, X, Y, Z): ##M*1,M*D,M*1, M*D
        """
        Compute the phi term in the differential equation.
        """
        return self.r*(Y-torch.sum(X*Z,1,keepdims=True))
    

    
    def sig(self, t, X, Y):
        """
        Compute the volatility matrix.
        """
        return self.sigma*torch.diag_embed(X) #M*D*D



class toy_example2(FBSDNN):
    def __init__(self, Xi, T, M, N, D, layers, learning_rate):
        super().__init__(Xi, T, M, N, D, layers, learning_rate)
   
    def u_exact(self,t,X): #K*1, K*D
        """
        real u
        """
        
        MC = 10**5
        K = t.shape[0]
        
        W = np.random.normal(size=(MC,K,self.D)) 
        W = torch.from_numpy(W)
        W = W.to(device)
        
        return -torch.log(torch.mean(torch.exp(-self.g_tc(X + torch.sqrt(2.0*torch.abs(self.T-t))*W)),axis=0))
    
    
   
    def g_tc(self, x):
        """
        Terminal condition function g(X_T).
        """
        return torch.log(0.5 + 0.5 * torch.sum(x**2, -1, keepdims=True))

   
    def phi(self, t, X, Y, Z):
        """
        Compute the phi term in the differential equation.
        """
        return torch.sum(Z**2, dim=1, keepdims=True)

    
    def sig(self, t, X, Y):
        """
        Compute the volatility matrix.
        """
        size = X.shape
        return np.sqrt(2) * torch.diag_embed(torch.ones(size)).to(device)
