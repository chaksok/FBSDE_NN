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
from torch.func import jacrev, vmap
sns.set()
# plt.style.use('ggplot')

torch.manual_seed(1)
np.random.seed(1)

PATH = "tryexact.pt"
# PATH = "tryexact2.pt"
# PATH = "tryexact3.pt"
device = 'cpu'

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
    def __init__(self,T,K,N,D,I,V0,layers,drift_D,sigD,muE,sigE,alpha,epsilon):
        self.T=T  #terminal time
        self.K=K   #number of trajectories
        self.N=N   #number of time snapshots
        self.D=D   #number of dimensions
        self.I=I   #number of agents
        
        self.V0=V0 #initial share of stock for each agent
        self.layers=layers   ## total layers of the NN
        self.drift_D=drift_D   ## coef of D=g(t,X) with respect to t
        self.sigD=sigD  ##coef of D=g(t,X) with respect to X
        self.muE=muE    ## coef of E=f_i(t,X) with respect to t
        self.sigE=sigE   ## coef of E=f_i(t,X) with respect to X
        self.alpha=alpha
        self.epsilon=epsilon
        self.model=myNN(self.layers).to(device)
        self.r_real={}
        self.t_star,self.W_star = self.gen_data(self.K, self.N, self.D, self.T)
        self.X={}
        
    
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

    
    # def net_u(self, t, x):
    #     Ii = self.I + 1
    #     # 建议：算雅可比时把 batch 依赖层设为 eval()，训练前再切回 train()
    #     # self.model.eval()

    #     # 单样本函数：输入 (D,), (t_dim,)；输出 (Ii,)
    #     def f(x_single, t_single):
    #         out = self.model(t_single.unsqueeze(0), x_single.unsqueeze(0))  # (1, Ii)
    #         return out.squeeze(0)                                           # (Ii,)

    #     # 逐样本雅可比 J: (B, Ii, D)，注意参数顺序传 (x, t)
    #     J = vmap(jacrev(f, argnums=0), in_dims=(0, 0))(x, t)

    #     # 批量前向
    #     y = self.model(t, x)                                 # (B, Ii)

    #     # 扁平化成你需要的形状
    #     z = J.reshape(x.size(0), Ii * self.D)                # (B, Ii*D)
    #     return y, z
    

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
    
   
    
    
    
    def sig(self,t,X): #M*1,M*D,M*1, volatility of state process X
        size=X.shape
        a=torch.ones(size).to(device)
        return torch.diag_embed(a) #M*D*D
    
   
    def S_exact(self,t,X): #K*1, K*D
        I=self.I
        sigD=self.sigD
        temp=self.sigD
        alpha=self.alpha
        for i in range(I):
            temp=temp+alpha[i]*self.sigE[i,:]
        a=torch.sum(temp*sigD,-1,keepdims=True)
        return (t-1)*a+torch.sum(sigD*X, -1, keepdims=True) #K*1
    
    def Y_exact(self, t, X, i):
        I=self.I
        alpha=self.alpha
        D=self.D
        sigD=self.sigD
        
        
        part2=0.5*torch.sum(self.sigE[i,:]**2,-1,keepdims=True)
        
        unit=sigD/torch.sqrt(torch.sum(sigD**2,-1,keepdims=True))
        temp=sigD
        for j in range(I):
            temp=temp+alpha[j]*self.sigE[j,:]
        part1=torch.sum((temp-self.sigE[i,:])*unit,-1,keepdims=True)
        a=part2-0.5*part1**2  ##M*1
        return (t-1)*a+torch.sum(self.sigE[i,:]*X, -1, keepdims=True)
    
    
    
    
    
    def predict(self):
        self.model.load_state_dict(torch.load(PATH)) ###reload the best model
        
        V0=self.V0
        I=self.I
        
        t_star,W_star = self.t_star,self.W_star
        
        t0=t_star[:,0,:].float() #K*1 initial time
        W0=W_star[:,0,:].float() #K*D
        
        
        Y_path=[]
        for i in range(I+1): #S,Y,h
            Y_path.append([])
        
        Y_real=[]
        for i in range(I+1): #S,Y,h
            Y_real.append([])
        
        
        Y0,Z0=self.net_u(t0,W0)
        S_0=self.S_exact(t0,W0)
        
        Y_real[0].append(S_0)
        
        for i in range(I+1):
            Y_path[i].append(Y0[:,i].unsqueeze(1))
            
        for i in range(1,I+1):
            Y_real[i].append(self.Y_exact(t0,W0,i-1).unsqueeze(1))
        
        for time in range(1,self.N+1): #iterate from time 1 to time N
            t1=t_star[:,time,:].float() #M*1
            W1=W_star[:,time,:].float() #M*1
            
            Y1, Z1=self.net_u(t1,W1)
            S_1=self.S_exact(t1,W1)
                
            for j in range(I+1):
                Y_path[j].append(Y1[:,j].unsqueeze(1))
            
            Y_real[0].append(S_1)
            
            for i in range(1,I+1):
                Y_real[i].append(self.Y_exact(t1,W1,i-1).unsqueeze(1))
            
            Y0=Y1
               
                
            t0=t1
            W0=W1
            
        
        
        for i in range(I+1):
            Y_path[i]=torch.stack(Y_path[i],dim=1)
        
        for i in range(I+1):
            Y_real[i]=torch.stack(Y_real[i],dim=1)
        
        
        return Y_path, Y_real
        
        
        
        
        
T=1 #terminal time
M=100  #number of trajectoris
N=100  #number of time step
D=4
I=3

K=5000
layers=[D+1]+4*[256]+[(I+1)*(1+D)]  #NN layers
# layers=[D+1]+5*[256]+[(I+1)]  #NN layers
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
torch.manual_seed(1)
np.random.seed(1)

myFBSDE=FBSDNN(T,K,N,D,I,V0,layers,drift_D,sigD,muE,sigE,alpha,epsilon)

Y_path,Y_real=myFBSDE.predict()

colour=['b','g','r','c','m','y','k','w','mediumseagreen','aquamarine']


S_star_path=Y_path[0]
Y1_star_path=Y_path[1]
Y2_star_path=Y_path[2]
Y3_star_path=Y_path[3]

S=Y_real[0]
R1=Y_real[1]
R2=Y_real[2]
R3=Y_real[3]

S_star_path=S_star_path.cpu()
Y1_star_path=Y1_star_path.cpu()
Y2_star_path=Y2_star_path.cpu()
Y3_star_path=Y3_star_path.cpu()

S=S.cpu()
R1=R1.cpu()
R2=R2.cpu()
R3=R3.cpu()


S_star_path=S_star_path.detach().numpy()
Y1_star_path=Y1_star_path.detach().numpy()
Y2_star_path=Y2_star_path.detach().numpy()
Y3_star_path=Y3_star_path.detach().numpy()

S=S.detach().numpy()
R1=R1.detach().numpy()
R2=R2.detach().numpy()
R3=R3.detach().numpy()


S_star_path=S_star_path.squeeze()
Y1_star_path=Y1_star_path.squeeze()
Y2_star_path=Y2_star_path.squeeze()
Y3_star_path=Y3_star_path.squeeze()

S=S.squeeze()
R1=R1.squeeze()
R2=R2.squeeze()
R3=R3.squeeze()

plt.figure(figsize=(16,10))
plt.subplot(2, 2, 1)
sns.set_style("white")
plt.plot(S_star_path[0,:],color='#2A77AC', linewidth=2, label=f'Learned $U^0(t, X_t)$')
plt.plot(S[0,:],color='#C9244C',linestyle='-.',linewidth=2,label=f'Exact $u^0(t, X_t)$')
plt.scatter(len(S_star_path[0]) - 1, S_star_path[0, -1], marker='s', s=60, color='black', label='$U^0(0, \Gamma_0)$')
plt.scatter(0, S_star_path[1, 0], marker='o', s=100, color='black', label='$U^0(1, X_1)$')

for i in range(1,5):
    plt.plot(S_star_path[i,:],color='#2A77AC', linewidth=2)
    plt.scatter(len(S_star_path[0]) - 1, S_star_path[i, -1], marker='s', s=60, color='black')
    plt.scatter(0, S_star_path[i, 0], marker='o', s=150, color='black')
    plt.plot(S[i,:],color='#C9244C',linestyle='-.',linewidth=2.)
plt.title('Price Process of Risky Asset S', fontsize=20)    

plt.ylabel("$S_t=U^0(t, X_t)$", fontsize=18)

plt.xticks(range(0,101,10),np.round(np.linspace(0.0, 1.0, num=11),decimals=1))
#plt.xlabel("t", fontsize=18)


plt.legend(fontsize=8,loc='upper left')
plt.subplot(2, 2, 2)
sns.set_style("white")
plt.plot(Y1_star_path[0,:],color='#2A77AC', linewidth=2, label=f'Learned $U^1(t, X_t)$')
plt.plot(R1[0,:],color='#C9244C',linestyle='-.',linewidth=2.,label=f'Exact $u^1(t, X_t)$')
plt.scatter(len(Y1_star_path[0]) - 1, Y1_star_path[0, -1], marker='s', s=60, color='black', label='$U^1(0, X_0)$')
plt.scatter(0, Y1_star_path[1, 0], marker='o', s=100, color='black', label='$U^1(1, X_1)$')

for i in range(1,5):
    plt.plot(Y1_star_path[i,:],color='#2A77AC', linewidth=2)
    plt.scatter(len(Y1_star_path[0]) - 1, Y1_star_path[i, -1], marker='s', s=60, color='black')
    plt.scatter(0, Y1_star_path[i, 0], marker='o', s=150, color='black')
    plt.plot(R1[i,:], color='#C9244C',linestyle='-.',linewidth=2.)
plt.title('Certainty equivalent of agent 1', fontsize=20)    

plt.ylabel("$Y^1_t=U^1(t, X_t)$", fontsize=18)

plt.xticks(range(0,101,10),np.round(np.linspace(0.0, 1.0, num=11),decimals=1))
#plt.xlabel("t", fontsize=18)


plt.legend(fontsize=8,loc='upper left')

plt.subplot(2, 2, 3)

sns.set_style("white")
plt.plot(Y2_star_path[0,:],color='#2A77AC', linewidth=2, label=f'Learned $U^2(t, X_t)$')
plt.plot(R2[0,:],color='#C9244C',linestyle='-.',linewidth=2.,label=f'Exact $u^2(t, X_t)$')
plt.scatter(len(Y2_star_path[0]) - 1, Y2_star_path[0, -1], marker='s', s=60, color='black', label='$U^2(0, X_0)$')
plt.scatter(0, Y2_star_path[1, 0], marker='o', s=100, color='black', label='$U^2(1, X_1)$')

for i in range(1,5):
    plt.plot(Y2_star_path[i,:],color='#2A77AC', linewidth=2)
    plt.scatter(len(Y2_star_path[0]) - 1, Y2_star_path[i, -1], marker='s', s=60, color='black')
    plt.scatter(0, Y2_star_path[i, 0], marker='o', s=150, color='black')
    plt.plot(R2[i,:],color='#C9244C',linestyle='-.',linewidth=2.)
plt.title('Certainty equivalent of agent 2', fontsize=20)    

plt.ylabel("$Y^2_t=U^2(t, X_t)$", fontsize=18)

plt.xticks(range(0,101,10),np.round(np.linspace(0.0, 1.0, num=11),decimals=1))
plt.xlabel("t", fontsize=18)


plt.legend(fontsize=8,loc='upper left')


plt.subplot(2, 2, 4)

sns.set_style("white")
plt.plot(Y3_star_path[0,:],color='#2A77AC', linewidth=2, label=f'Learned $U^3(t, X_t)$')
plt.plot(R3[0,:],color='#C9244C',linestyle='-.',linewidth=2.,label=f'Exact $u^3(t, X_t)$')
plt.scatter(len(Y3_star_path[0]) - 1, Y3_star_path[0, -1], marker='s', s=60, color='black', label='$U^3(0, X_0)$')
plt.scatter(0, Y3_star_path[1, 0], marker='o', s=100, color='black', label='$U^3(1, X_1)$')

for i in range(1,5):
    plt.plot(Y3_star_path[i,:],color='#2A77AC', linewidth=2)
    plt.scatter(len(Y3_star_path[0]) - 1, Y3_star_path[i, -1], marker='s', s=60, color='black')
    plt.scatter(0, Y3_star_path[i, 0], marker='o', s=150, color='black')
    plt.plot(R3[i,:],color='#C9244C',linestyle='-.',linewidth=2.)
plt.title('Certainty equivalent of agent 3', fontsize=20)    

plt.ylabel("$Y^3_t=U^3(t, X_t)$", fontsize=18)

plt.xticks(range(0,101,10),np.round(np.linspace(0.0, 1.0, num=11),decimals=1))
plt.xlabel("t", fontsize=18)


plt.legend(fontsize=8,loc='upper left')



plt.savefig('/Users/sokchak/Desktop/pic/path.pdf')
# plt.savefig('/Users/sokchak/Desktop/pic/path2.pdf')
# plt.savefig('/Users/sokchak/Desktop/pic/path3.pdf')
#plt.savefig('/Users/shuozhai/Desktop/pic/Y0path.pdf')


plt.figure(figsize=(16,10))
plt.subplot(2, 2, 1)
error=np.sqrt((S_star_path-S)**2/(np.maximum(zeta**2, S**2)))
mean_errors=np.mean(error,0)
std_error=np.std(error,0)

plt.plot(mean_errors,'#1B7C3D',label='mean')
plt.plot(mean_errors+2*std_error,'#F16C23',linestyle='--',label='mean+two std', linewidth=2)

plt.xticks(range(0,101,10),np.round(np.linspace(0.0, 1.0, num=11),decimals=1))
#plt.xlabel("t", fontsize=18)
plt.ylabel("relative error", fontsize=18)
plt.legend(fontsize=14,loc='upper right')
plt.title('Price Process of Risky Asset S', fontsize=20)    


plt.subplot(2, 2, 2)
error=np.sqrt((Y1_star_path-R1)**2/(np.maximum(zeta**2, R1**2)))
mean_errors=np.mean(error,0)
std_error=np.std(error,0)

plt.plot(mean_errors,'#1B7C3D',label='mean')
plt.plot(mean_errors+2*std_error,'#F16C23',linestyle='--',label='mean+two std', linewidth=2)

plt.xticks(range(0,101,10),np.round(np.linspace(0.0, 1.0, num=11),decimals=1))
#plt.xlabel("t", fontsize=18)
#plt.ylabel("relative error", fontsize=18)
plt.legend(fontsize=14,loc='upper right')
plt.title('Certainty equivalent of agent 1', fontsize=20)    

plt.subplot(2, 2, 3)
error=np.sqrt((Y2_star_path-R2)**2/(np.maximum(zeta**2, R2**2)))
mean_errors=np.mean(error,0)
std_error=np.std(error,0)

plt.plot(mean_errors,'#1B7C3D',label='mean')
plt.plot(mean_errors+2*std_error,'#F16C23',linestyle='--',label='mean+two std', linewidth=2)

plt.xticks(range(0,101,10),np.round(np.linspace(0.0, 1.0, num=11),decimals=1))
plt.xlabel("t", fontsize=18)
plt.ylabel("relative error", fontsize=18)
plt.legend(fontsize=14,loc='upper right')
plt.title('Certainty equivalent of agent 2', fontsize=20)    

plt.subplot(2, 2, 4)
error=np.sqrt((Y3_star_path-R3)**2/(np.maximum(zeta**2, R3**2)))
mean_errors=np.mean(error,0)
std_error=np.std(error,0)

plt.plot(mean_errors,'#1B7C3D',label='mean')
plt.plot(mean_errors+2*std_error,'#F16C23',linestyle='--',label='mean+two std', linewidth=2)

plt.xticks(range(0,101,10),np.round(np.linspace(0.0, 1.0, num=11),decimals=1))
plt.xlabel("t", fontsize=18)
#plt.ylabel("relative error", fontsize=18)
plt.legend(fontsize=14,loc='upper right')
plt.title('Certainty equivalent of agent 3', fontsize=20)    
plt.savefig('/Users/sokchak/Desktop/pic/error.pdf')
# plt.savefig('/Users/sokchak/Desktop/pic/error2.pdf')
# plt.savefig('/Users/sokchak/Desktop/pic/error3.pdf')