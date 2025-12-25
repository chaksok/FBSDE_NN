import torch
import pandas as pd
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

device='cpu'

class myNN(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.linear_gelu_stack = nn.Sequential()
        for i in range(len(layers)-2):
            self.linear_gelu_stack.add_module(f"linear{i}",nn.Linear(layers[i], layers[i+1]))
            self.linear_gelu_stack.add_module(f'activation{i}',nn.GELU())                                   
        self.linear_gelu_stack.add_module("flinear",nn.Linear(layers[len(layers)-2],layers[len(layers)-1]))

    def forward(self,t,x):
        results = self.linear_gelu_stack(torch.cat((t,x),1))
        return results

class FBSDNN(ABC):
    def __init__(self,T,M,N,D,I,K,layers,drift_D,sigD,muE,sigE,alpha,epsilon):
      
        self.T=T  #terminal time
        self.M=M   #number of trajectories
        self.N=N   #number of time snapshots
        self.D=D   #number of dimensions
        self.I=I   #number of agents
        self.K=K
        
        self.layers=layers   ## total layers of the NN
        self.drift_D=drift_D   ## coef of D=g(t,X) with respect to t
        self.sigD=sigD  ##coef of D=g(t,X) with respect to X
        self.muE=muE    ## coef of E=f_i(t,X) with respect to t
        self.sigE=sigE   ## coef of E=f_i(t,X) with respect to X
        self.alpha=alpha
        self.epsilon=epsilon
        self.model=myNN(self.layers).to(device)
        self.valid_error=[]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.t_valid,self.W_valid=self.gen_data(self.K,self.N,self.D,self.T)
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.train_loss = []
    
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
    def g_Di(self,t,X): #dividend process 
        pass  #M*1
    
    
    @abstractmethod
    def g_E(self,t,X,i): #endowment process for agent i 
        pass  
    
    
    
   
    
    @abstractmethod
    def phi0(self,t,X,Y,Z,Di,epsilon): #drift for S
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
  
  
   
    def train(self,NIter,epoch,learning_rate,patience):
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
                self.train_loss.append(loss.item())
                if (i+1)%200==0:
                    elapsed=time.time()-start_time
                    print(f"Loss {(i+1)}: {loss.item()}, time: {elapsed}")
                    
            val_loss=(self.loss_func(self.t_valid,self.W_valid,self.K)).item()
            print(f"Validation Loss {(j+1)}: {val_loss:.4f}")
            self.valid_error.append(val_loss)
            # Early stopping and model checkpointing
            if val_loss < self.best_val_loss:
                print(f"val_loss: {val_loss:.4f}, updating best_val_loss from: {self.best_val_loss:.4f}.")
                self.best_val_loss = val_loss
                self.best_epoch = j
                torch.save(self.model.state_dict(),PATH)
                early_stopping_counter = 0
            else:
                print(f"val_loss: {val_loss:.4f}, worse than best_val_loss: {self.best_val_loss:.4f}.")
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {j+1}")
                    break
        
        self.model.load_state_dict(torch.load(PATH)) ###reload the best model
        
    
    @abstractmethod
    def theta(self,Y,Z,K):
        pass
    
    
    
class B_S_B2(FBSDNN):
    def __init__(self,T,M,N,D,I,K,layers,drift_D,sigD,muE,sigE,alpha,epsilon):
        super().__init__(T,M,N,D,I,K,layers,drift_D,sigD,muE,sigE,alpha,epsilon)
    
    
    def g_Di(self,t,X):  #dividend process Di=g(t,X)
        mu=self.drift_D
        sig=self.sigD
        if len(sig[0,:])!=self.D:
            raise ValueError(f"the volatility of the asset should have {self.D} elements.\n")
        return torch.sum(sig*X,-1,keepdims=True)
    
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
        


class FBSDNN(ABC):
    def __init__(self,T,M,N,D,I,K,layers,drift_D,sigD,muE,sigE,alpha,epsilon):
      
        self.T=T  #terminal time
        self.M=M   #number of trajectories
        self.N=N   #number of time snapshots
        self.D=D   #number of dimensions
        self.I=I   #number of agents
        self.K=K
        
        self.layers=layers   ## total layers of the NN
        self.drift_D=drift_D   ## coef of D=g(t,X) with respect to t
        self.sigD=sigD  ##coef of D=g(t,X) with respect to X
        self.muE=muE    ## coef of E=f_i(t,X) with respect to t
        self.sigE=sigE   ## coef of E=f_i(t,X) with respect to X
        self.alpha=alpha
        self.epsilon=epsilon
        self.model=myNN(self.layers).to(device)
        self.valid_error=[]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.t_valid,self.W_valid=self.gen_data(self.K,self.N,self.D,self.T)
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.train_loss = []
    
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
    
    # def net_u(self,t,x): 
    #     Ii=self.I+1
    #     x.requires_grad = True
    #     result=self.model(t,x)
    #     y = result[:,0:Ii]   #M*(I*1)
    #     z = result[:,Ii:Ii*(D+1)] #M*(I*D)
    #     return y,z
    
    

    def net_u(self, t, x):
        Ii = self.I + 1
        # 建议：算雅可比时把 batch 依赖层设为 eval()，训练前再切回 train()
        # self.model.eval()

        # 单样本函数：输入 (D,), (t_dim,)；输出 (Ii,)
        def f(x_single, t_single):
            out = self.model(t_single.unsqueeze(0), x_single.unsqueeze(0))  # (1, Ii)
            return out.squeeze(0)                                           # (Ii,)

        # 逐样本雅可比 J: (B, Ii, D)，注意参数顺序传 (x, t)
        J = vmap(jacrev(f, argnums=0), in_dims=(0, 0))(x, t)

        # 批量前向
        y = self.model(t, x)                                 # (B, Ii)

        # 扁平化成你需要的形状
        z = J.reshape(x.size(0), Ii * self.D)                # (B, Ii*D)
        return y, z
    

    # def bsde_2p8_terms_hutch(self, t, x, dW, h, s=2, create_graph=True):
    # # """
    # # 近似计算每个 U_k 的：
    # #   - μ^T H_k σ h ΔW  - 1/2 ΔW^T σ^T H_k σ ΔW + 1/2 tr(σ^T H_k σ) h
    # # 用 Hutchinson 把 tr(.) 从 D 次 HVP 降到 s 次 HVP。
    # # 输入:
    # #   t: (B, tdim)
    # #   x: (B, D)
    # #   dW: (B, D)
    # #   h:  () 或 (B,1)
    # #   s:  Hutchinson 采样数，1~4 通常足够
    # # 返回:
    # #   expr: (B, K)
    # # """
    #     B, D = x.shape
    #     x = x.clone().detach().requires_grad_(True)

    #     U   = self.model(t, x)                     # (B, K)
    #     K   = U.shape[1]
    #     mu  = self.mu(t, x)                        # (B, D)
    #     Sig = self.sig(t, x)                       # (B, D, D)

    #     # v = σ ΔW
    #     v = torch.einsum('bdd,bd->bd', Sig, dW)    # (B, D)

    #     if not torch.is_tensor(h):
    #         h = torch.tensor(h, device=x.device, dtype=x.dtype)
    #     if h.dim() == 0:
    #         h = h.expand(B, 1)                     # (B,1)

    #     def hvp_from_grad(gk, v_):
    #         # 二次求导的 HVP：H v = d/dx <∇U_k, v>
    #         return torch.autograd.grad(
    #             outputs=(gk * v_).sum(), inputs=x,
    #             retain_graph=True, create_graph=create_graph, only_inputs=True
    #         )[0]                                    # (B, D)

    #     out_terms = []
    #     for k in range(K):
    #         # ∇_x U_k
    #         gk = torch.autograd.grad(
    #             outputs=U[:, k].sum(), inputs=x,
    #             retain_graph=True, create_graph=True, only_inputs=True
    #         )[0]                                     # (B, D)

    #         # ① HσΔW
    #         Hv = hvp_from_grad(gk, v)                # (B, D)
    #         term1 = - (h * (mu * Hv).sum(dim=1, keepdim=True))        # (B,1)

    #         # ② v^T H v
    #         term2 = -0.5 * (Hv * v).sum(dim=1, keepdim=True)          # (B,1)

    #         # ③ 迹项：Hutchinson 估计  (1/s) Σ_r (H v_z)·v_z,  v_z = σ z_r
    #         tr_acc = x.new_zeros(B, 1)
    #         for _ in range(s):
    #             # Rademacher 向量 z ∈ {±1}^D
    #             z = torch.empty(B, D, device=x.device, dtype=x.dtype).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    #             vz = torch.einsum('bdd,bd->bd', Sig, z)              # v_z = σ z
    #             Hvz = hvp_from_grad(gk, vz)                          # (B, D)
    #             tr_acc = tr_acc + (Hvz * vz).sum(dim=1, keepdim=True)
    #         term3 = 0.5 * h * (tr_acc / float(s))                    # (B,1)

    #         out_terms.append(term1 + term2 + term3)                  # (B,1)

    #     expr = torch.cat(out_terms, dim=1)                           # (B, K)
    #     return expr


    # def net_u(self, t, x, create_graph=True, flatten=True):
    #     """
    #     返回:
    #     y : (B, I+1)
    #     z : (B, (I+1)*D)  若 flatten=False 则返回 (B, I+1, D)
    #     计算方式：直接求 JVP 得到 Z = (∇_x y) @ sigma(t,x)，避免显式构建整张雅可比；
    #     若 jvp 不可用则回退到反向模式循环 + einsum。
    #     """
    #     Ii = self.I + 1
    #     t = t.detach()
    #     x = x.clone().detach().requires_grad_(True)
    #     B, D = x.shape

    #     # 前向
    #     y = self.model(t, x)  # (B, Ii)

    #     # σ(t,x): (B, D, D)
    #     Sigma = self.sig(t, x)

    #     try:
    #         # --- 快速路径：JVP (需要 PyTorch 2.x 的 torch.func) ---
    #         from torch.func import jvp, vmap

    #         # 单样本函数：(D,)->(Ii,)
    #         def f_single(x1, t1):
    #             return self.model(t1.unsqueeze(0), x1.unsqueeze(0)).squeeze(0)

    #         # 对单样本计算 (∇y) @ Sigma ：按列做 jvp 得到 J v，再拼成 (Ii, D)
    #         def J_times_Sigma_single(x1, t1, S1):
    #             def jvp_col(v):
    #                 _, jvp_val = jvp(lambda xx: f_single(xx, t1), (x1,), (v,))
    #                 return jvp_val  # (Ii,)
    #             return vmap(jvp_col, in_dims=0)(S1.T).T  # (Ii, D)

    #         # 批量化：(B, Ii, D)
    #         Z_blocks = vmap(J_times_Sigma_single, in_dims=(0, 0, 0))(x, t, Sigma)

    #     except Exception:
    #         # --- 回退路径：反向模式循环 Ii 次 + 批量乘 σ ---
    #         grads = []
    #         for k in range(Ii):
    #             gk = torch.autograd.grad(
    #                 outputs=y[:, k].sum(), inputs=x,
    #                 retain_graph=(k < Ii - 1),
    #                 create_graph=create_graph,      # 需要通过 z 回传到参数时为 True
    #                 only_inputs=True,
    #             )[0]                                # (B, D)
    #             grads.append(gk)
    #         J_blocks = torch.stack(grads, dim=1)    # (B, Ii, D)
    #         Z_blocks = torch.einsum('bid,bdd->bid', J_blocks, Sigma)  # (B, Ii, D)

    #     if flatten:
    #         z = Z_blocks.reshape(B, Ii * D)         # (B, (I+1)*D)
    #     else:
    #         z = Z_blocks                             # (B, I+1, D)

    #     if not create_graph:
    #         z = z.detach()
    #     return y, z
    


    @abstractmethod
    def g_Di(self,t,X): #dividend process 
        pass  #M*1
    
    
    @abstractmethod
    def g_E(self,t,X,i): #endowment process for agent i 
        pass  
    
    
    
   
    
    @abstractmethod
    def phi0(self,t,X,Y,Z,Di,epsilon): #drift for S
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
        I=self.I
        loss=torch.tensor([0.0],device=device)

        
        t0=t[:,0,:].float() #M*1 initial time
        W0=W[:,0,:].float() #M*D

        
            
        
        
        Y0,Z0=self.net_u(t0,W0)  #M*(I*1), M*(I*D)
        for i in range(I+1):
            Z0[:,i*D:(i+1)*D]=(Z0[:,i*D:(i+1)*D].unsqueeze(1)@self.sig(t0,W0)).squeeze()

        
        for time in range(1,self.N+1): #iterate from time 1 to time N
            t1=t[:,time,:].float() #M*1
            W1=W[:,time,:].float() #M*1
            
            # expr = self.bsde_2p8_terms_hutch(t0, W0, W1 - W0, t1 - t0)
            
            Y0_0=Y0[:,0].unsqueeze(1)
            Z0_0=Z0[:,0:D] 
            
            Y1_temp=[]
            Y0_1_pred=Y0_0+self.phi0(t0,W0,Z0,epsilon)*(t1-t0)+torch.sum(Z0_0*(W1-W0),1,keepdims=True) #+ expr[:, 0].unsqueeze(1)
            Y1_temp.append(Y0_1_pred)
            for j in range(1,I+1):
                pred=Y0[:,j].unsqueeze(1)+self.phi(t0,W0,Z0,j,epsilon)*(t1-t0)+torch.sum(Z0[:,j*D:(j+1)*D]*(W1-W0),1,keepdims=True) #+ expr[:, i].unsqueeze(1)
                Y1_temp.append(pred)
            
            
        
            Y1_pred=torch.cat(Y1_temp,1)  #M*(I+2)
            
            Y1,Z1=self.net_u(t1,W1)
            for i in range(I+1):
                Z1[:,i*D:(i+1)*D]=(Z1[:,i*D:(i+1)*D].unsqueeze(1)@self.sig(t1,W1)).squeeze()
        
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

        loss += nn.MSELoss(reduction='mean')(Y1,Y_terminal)
        # loss+=0.05*nn.MSELoss(reduction='mean')(Z1,Z_terminal)
       
        
        return loss
  
  
   
    def train(self,NIter,epoch,learning_rate,patience):
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
                self.train_loss.append(loss.item())
                if (i+1)%200==0:
                    elapsed=time.time()-start_time
                    print(f"Loss {(i+1)}: {loss.item()}, time: {elapsed}")
                    
            val_loss=(self.loss_func(self.t_valid,self.W_valid,self.K)).item()
            print(f"Validation Loss {(j+1)}: {val_loss:.4f}")
            self.valid_error.append(val_loss)
            # Early stopping and model checkpointing
            if val_loss < self.best_val_loss:
                print(f"val_loss: {val_loss:.4f}, updating best_val_loss from: {self.best_val_loss:.4f}.")
                self.best_val_loss = val_loss
                self.best_epoch = j
                torch.save(self.model.state_dict(),PATH)
                early_stopping_counter = 0
            else:
                print(f"val_loss: {val_loss:.4f}, worse than best_val_loss: {self.best_val_loss:.4f}.")
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {j+1}")
                    break
        
        self.model.load_state_dict(torch.load(PATH)) ###reload the best model
        
    
    @abstractmethod
    def theta(self,Y,Z,K):
        pass
    
    
    
class B_S_B2(FBSDNN):
    def __init__(self,T,M,N,D,I,K,layers,drift_D,sigD,muE,sigE,alpha,epsilon):
        super().__init__(T,M,N,D,I,K,layers,drift_D,sigD,muE,sigE,alpha,epsilon)
    
    
    def g_Di(self,t,X):  #dividend process Di=g(t,X)
        mu=self.drift_D
        sig=self.sigD
        if len(sig[0,:])!=self.D:
            raise ValueError(f"the volatility of the asset should have {self.D} elements.\n")
        return torch.sum(sig*X,-1,keepdims=True)
    
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
                tmp=tmp+alpha[i]*Z[:,(i+1)*D:(i+2)*D]

            for i in range(I):
                theta_i=1+torch.sum((tmp-Z[:,(i+1)*D:(i+2)*D])*temp,-1,keepdims=True)
                theta.append(theta_i)
    
            return theta
    

class FBSDNN(ABC):
    def __init__(self,T,M,N,D,I,K,layers,drift_D,sigD,muE,sigE,alpha,epsilon):
      
        self.T=T  #terminal time
        self.M=M   #number of trajectories
        self.N=N   #number of time snapshots
        self.D=D   #number of dimensions
        self.I=I   #number of agents
        self.K=K
        
        self.layers=layers   ## total layers of the NN
        self.drift_D=drift_D   ## coef of D=g(t,X) with respect to t
        self.sigD=sigD  ##coef of D=g(t,X) with respect to X
        self.muE=muE    ## coef of E=f_i(t,X) with respect to t
        self.sigE=sigE   ## coef of E=f_i(t,X) with respect to X
        self.alpha=alpha
        self.epsilon=epsilon
        self.model=myNN(self.layers).to(device)
        self.valid_error=[]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.t_valid,self.W_valid=self.gen_data(self.K,self.N,self.D,self.T)
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.train_loss = []
    
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
    
    def bsde_2p8_terms_ultrafast(self, t, x, dW, h):
        """
        超快版本：跳过迹项，只算主要的两项
        适合快速原型和调试
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
            # 跳过 term3（迹项）
            
            out[:, k] = (term1 + term2).squeeze(1)
        
        return out



    # === 整合版本：修改 net_u 以配合使用 ===
    def net_u(self, t, x):
        Ii = self.I + 1
        # 建议：算雅可比时把 batch 依赖层设为 eval()，训练前再切回 train()
        # self.model.eval()

        # 单样本函数：输入 (D,), (t_dim,)；输出 (Ii,)
        def f(x_single, t_single):
            out = self.model(t_single.unsqueeze(0), x_single.unsqueeze(0))  # (1, Ii)
            return out.squeeze(0)                                           # (Ii,)

        # 逐样本雅可比 J: (B, Ii, D)，注意参数顺序传 (x, t)
        J = vmap(jacrev(f, argnums=0), in_dims=(0, 0))(x, t)

        # 批量前向
        y = self.model(t, x)                                 # (B, Ii)

        # 扁平化成你需要的形状
        z = J.reshape(x.size(0), Ii * self.D)                # (B, Ii*D)
        return y, z

    # === 使用示例 ===
    

    @abstractmethod
    def g_Di(self,t,X): #dividend process 
        pass  #M*1
    
    
    @abstractmethod
    def g_E(self,t,X,i): #endowment process for agent i 
        pass  
    
    
    
   
    
    @abstractmethod
    def phi0(self,t,X,Y,Z,Di,epsilon): #drift for S
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
        I=self.I
        loss=torch.tensor([0.0],device=device)

        
        t0=t[:,0,:].float() #M*1 initial time
        W0=W[:,0,:].float() #M*D

        
            
        
        
        Y0,Z0 =self.net_u(t0,W0)  #M*(I*1), M*(I*D)
        for i in range(I+1):
            Z0[:,i*D:(i+1)*D]=(Z0[:,i*D:(i+1)*D].unsqueeze(1)@self.sig(t0,W0)).squeeze()

        
        for time in range(1,self.N+1): #iterate from time 1 to time N
            t1=t[:,time,:].float() #M*1
            W1=W[:,time,:].float() #M*1
            
            expr = self.bsde_2p8_terms_ultrafast(t0, W0, W1 - W0, t1 - t0)
            
            Y0_0=Y0[:,0].unsqueeze(1)
            Z0_0=Z0[:,0:D] 
            
            Y1_temp=[]
            Y0_1_pred=Y0_0+self.phi0(t0,W0,Z0,epsilon)*(t1-t0)+torch.sum(Z0_0*(W1-W0),1,keepdims=True) + expr[:, 0].unsqueeze(1)
            Y1_temp.append(Y0_1_pred)
            for j in range(1,I+1):
                pred=Y0[:,j].unsqueeze(1)+self.phi(t0,W0,Z0,j,epsilon)*(t1-t0)+torch.sum(Z0[:,j*D:(j+1)*D]*(W1-W0),1,keepdims=True) + expr[:, i].unsqueeze(1)
                Y1_temp.append(pred)
            
            
        
            Y1_pred=torch.cat(Y1_temp,1)  #M*(I+2)
            
            Y1,Z1=self.net_u(t1,W1)
            for i in range(I+1):
                Z1[:,i*D:(i+1)*D]=(Z1[:,i*D:(i+1)*D].unsqueeze(1)@self.sig(t1,W1)).squeeze()
        
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

        loss += nn.MSELoss(reduction='mean')(Y1,Y_terminal)
        # loss+=0.05*nn.MSELoss(reduction='mean')(Z1,Z_terminal)
       
        
        return loss
  
  
   
    def train(self,NIter,epoch,learning_rate,patience):
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
                self.train_loss.append(loss.item())
                if (i+1)%200==0:
                    elapsed=time.time()-start_time
                    print(f"Loss {(i+1)}: {loss.item()}, time: {elapsed}")
                    
            val_loss=(self.loss_func(self.t_valid,self.W_valid,self.K)).item()
            print(f"Validation Loss {(j+1)}: {val_loss:.4f}")
            self.valid_error.append(val_loss)
            # Early stopping and model checkpointing
            if val_loss < self.best_val_loss:
                print(f"val_loss: {val_loss:.4f}, updating best_val_loss from: {self.best_val_loss:.4f}.")
                self.best_val_loss = val_loss
                self.best_epoch = j
                torch.save(self.model.state_dict(),PATH)
                early_stopping_counter = 0
            else:
                print(f"val_loss: {val_loss:.4f}, worse than best_val_loss: {self.best_val_loss:.4f}.")
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping at epoch {j+1}")
                    break
        
        self.model.load_state_dict(torch.load(PATH)) ###reload the best model
        
    
    @abstractmethod
    def theta(self,Y,Z,K):
        pass
    
    
    
class B_S_B2(FBSDNN):
    def __init__(self,T,M,N,D,I,K,layers,drift_D,sigD,muE,sigE,alpha,epsilon):
        super().__init__(T,M,N,D,I,K,layers,drift_D,sigD,muE,sigE,alpha,epsilon)
    
    
    def g_Di(self,t,X):  #dividend process Di=g(t,X)
        mu=self.drift_D
        sig=self.sigD
        if len(sig[0,:])!=self.D:
            raise ValueError(f"the volatility of the asset should have {self.D} elements.\n")
        return torch.sum(sig*X,-1,keepdims=True)
    
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
                tmp=tmp+alpha[i]*Z[:,(i+1)*D:(i+2)*D]

            for i in range(I):
                theta_i=1+torch.sum((tmp-Z[:,(i+1)*D:(i+2)*D])*temp,-1,keepdims=True)
                theta.append(theta_i)
    
            return theta
    


    
if __name__ == "__main__":
    T=1 #terminal time
    M=100  #number of trajectoris
    N=100  #number of time step
    D=4
    I=3

    K=1000
    layers=[D+1]+4*[256]+[(I+1)*(1+D)]  #NN layers
    drift_D=0.2
    sigD=torch.tensor([[0.3,0.,0.1,0.0]],device=device)
    muE=torch.tensor([0.1,0.1,0.1],device=device)
    sigE=torch.tensor([[0.3,0.3,0.,0.],[0.2,0.,0.3,0.],[0.1,0.,0.0,0.3]],device=device)
    alpha=[0.4,0.3,0.3]
    epsilon=1e-7
    learning_rate=1e-4
    V0=[1,1,1] #initial share of each agent, should satisfies alpha*V0=1
    epoch=50
    NIter=1000 #number of iteration
    patience=2

    myFBSDE=B_S_B2(T,M,N,D,I,K,layers,drift_D,sigD,muE,sigE,alpha,epsilon)
    myFBSDE.train(NIter,5,1e-4,patience)


    # T=1 #terminal time
    # M=100  #number of trajectoris
    # N=100  #number of time step
    # D=4
    # I=3

    # K=1000
    # layers=[D+1]+5*[256]+[(I+1)]  #NN layers
    # drift_D=0.2
    # sigD=torch.tensor([[0.3,0.,0.1,0.0]],device=device)
    # muE=torch.tensor([0.1,0.1,0.1],device=device)
    # sigE=torch.tensor([[0.3,0.3,0.,0.],[0.2,0.,0.3,0.],[0.1,0.,0.0,0.3]],device=device)
    # alpha=[0.4,0.3,0.3]
    # epsilon=1e-7
    # learning_rate=1e-4
    # V0=[1,1,1] #initial share of each agent, should satisfies alpha*V0=1
    # epoch=50
    # NIter=1000 #number of iteration
    # patience=2
    # torch.manual_seed(1)
    # np.random.seed(1)

    # T=1 #terminal time
    # M=100  #number of trajectoris
    # N=100  #number of time step
    # D=4
    # I=3

    # K=1000
    # layers=[D+1]+5*[256]+[(I+1)]  #NN layers
    # drift_D=0.2
    # sigD=torch.tensor([[0.3,0.,0.1,0.0]],device=device)
    # muE=torch.tensor([0.1,0.1,0.1],device=device)
    # sigE=torch.tensor([[0.3,0.3,0.,0.],[0.2,0.,0.3,0.],[0.1,0.,0.0,0.3]],device=device)
    # alpha=[0.4,0.3,0.3]
    # epsilon=1e-7
    # learning_rate=1e-4
    # V0=[1,1,1] #initial share of each agent, should satisfies alpha*V0=1
    # epoch=50
    # NIter=1000 #number of iteration
    # patience=2
    # torch.manual_seed(1)
    # np.random.seed(1)