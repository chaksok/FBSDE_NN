import torch
import numpy as np
import copy
import torch.nn.functional as F
from core.data_generator import BrownianMotionGenerator
from models.radner_terminal_muti_solver import RadnerEquilibriumSolverMulti4

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Training hyperparameters & Dimensions
# ---------------------------------------------------------------------------
epoch = 10       # Number of training epochs
NIter = 1000     # Number of iterations per epoch
patience = 2     # Patience for early stopping

T = 1.0
M = 128         # Batch size
N = 50          # Time steps
K = 2000         # Validation size

# # 多资产核心维度设定
# S = 2            # Number of stocks (M in the paper)
# I = 2            # Number of agents
# D = 3            # Number of Brownian motions (W1: Stock A, W2: Stock B, W3: Weather)

# print("Generating training data...")
# t_train, W_train = BrownianMotionGenerator.generate(M * NIter, N, D, T)
# t_valid, W_valid = BrownianMotionGenerator.generate(K, N, D, T)

# # ---------------------------------------------------------------------------
# # 2. Configuration Dictionary
# # ---------------------------------------------------------------------------
# config = {
#     'T': T,
#     'M': M,
#     'N': N,
#     'D': D,
#     'S': S,  # <-- 多资产新增参数
#     'I': I,
#     'K': K,
#     'learning_rate': 1e-4,
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
#     # 网络架构升级：Y输出 S+I 维， Z输出 (S+I)*D 维
#     'layers_y': [D + 1] + 5 * [256] + [S + I],
#     'layers_z': [D + 1] + 5 * [256] + [(S + I) * D],
    
#     'drift_D': 0.0,
    
#     # 股票波动率 (S=2, D=3): 股票A暴露于W1，股票B暴露于W2
#     'sigD': torch.tensor([
#         [0.24/np.sqrt(2), 0.0, 0.0],  # Stock A
#         [0.0, 0.24/np.sqrt(2), 0.0]   # Stock B
#     ], dtype=torch.float32),
    
#     'muE': torch.zeros(I),
    
#     # 禀赋波动率 (I=2, D=3): 两个代理人只暴露于天气风险 W3
#     'sigE': torch.tensor([
#         [0.0, 0.0, 0.6/np.sqrt(2)],  # Agent1 exposure to W3
#         [0.0, 0.0, 0.4/np.sqrt(2)]   # Agent2 exposure to W3
#     ], dtype=torch.float32),
    
#     'alpha': torch.tensor([0.4, 0.6]),  # CARA risk aversion
#     'epsilon': 1e-6,
    
#     'N_option': 1.0, 
#     'a_expo': 0.5,   
#     'x2': 0.0,       # Coefficient for Weather state shift
#     'checkpoint_path': 'radner_multi_asset_put.pth',
    
#     't_train': t_train,
#     'W_train': W_train,
#     't_valid': t_valid,
#     'W_valid': W_valid
# }

# ---------------------------------------------------------------------------
# 3. Custom Solver Class for Multi-Stock
# ---------------------------------------------------------------------------
class RadnerEquilibriumSolver_multi_put(RadnerEquilibriumSolverMulti4):
    def __init__(self, config):
        """Initialize multi-stock solver with specific payoff structure."""
        super().__init__(config)
        self.x2 = config['x2']
        
    def endowment_process(self, t, X, i):
        """
        Custom endowment process for multi-asset.
        """
        N_option = self.N_option
        a = self.a_expo
        x2 = self.x2

        # 提取天气风险变量: 通过与 sigE 的内积，自动抓取有暴露的布朗运动分量
        # 相比单资产，用 sum 更加安全，不用管是第几个 index
        X_shifted = X + x2
        X_weather = (self.sigE[i, :] * X_shifted).sum(dim=-1, keepdim=True)
    
        # 混合收益结构
        # payoff = (1 - a) * X_weather + a * torch.minimum(X_weather, torch.zeros_like(X_weather))
        payoff = X_weather

        if i == 0:
            return -N_option * payoff
        else:
            return N_option * payoff
    
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
    

    def get_analytical_theta(self):
        """
        Analytical Radner equilibrium strategies.
        Returns:
            Tensor of shape (I,) : theta^i
        """
        b0 = self.sigD.squeeze(0)                # (D,)
        b_agents = self.sigE # (I, D)

        # numerator common term
        sum_b = b0 + (self.alpha.unsqueeze(1) * b_agents).sum(0)
        common = torch.dot(sum_b, b0)

        denom = torch.dot(b0, b0)

        theta = (common - torch.mv(b_agents, b0)) / denom
        return theta  # (I,)
    

    

# # ---------------------------------------------------------------------------
# # 4. Initialization and Training execution
# # ---------------------------------------------------------------------------
# print("\nProblem Configuration:")
# print(f"  Terminal time: {config['T']}")
# print(f"  Brownian dimension (D): {config['D']}")
# print(f"  Number of Stocks (S): {config['S']}")
# print(f"  Number of agents (I): {config['I']}")
# print(f"  Time steps (N): {config['N']}")
# print(f"  Batch size (M): {config['M']}")
# print(f"  Y-Network architecture: {config['layers_y']}")
# print(f"  Z-Network architecture: {config['layers_z']}")

# # Initialize solver
# print("\nInitializing multi-asset solver...")
# solver = RadnerEquilibriumSolver_multi_put(config)

# print("\nStarting training...")
# solver.train(NIter, epoch, patience)     

# # Print training summary
# print("\n" + "="*60)
# print("Training Summary:")
# print("="*60)
# print(f"Best epoch: {solver.history['best_epoch'] + 1}")
# print(f"Best validation loss: {solver.history['best_val_loss']:.6f}")
# print(f"Final training loss: {solver.history['train_loss'][-1]:.6f}")
# print(f"Total training time: {solver.history['train_time']:.2f} seconds")
# print(f"Average time per epoch: {solver.history['time_per_epoch']:.2f} seconds")


S = 1  # 回归单股票
I = 2
D = 2  # 只需要二维布朗运动 (W1, W2)

# 重新生成数据
t_train, W_train = BrownianMotionGenerator.generate(M * NIter, N, D, T)
t_valid, W_valid = BrownianMotionGenerator.generate(K, N, D, T)

config_single = {
    'T': T, 'M': M, 'N': N, 'D': D, 
    'S': S, 'I': I, 'K': K,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 网络架构自动降维为 1+2 = 3维 和 3*2 = 6维
    'layers_y': [D + 1] + 5 * [256] + [S + I],
    'layers_z': [D + 1] + 5 * [256] + [(S + I) * D],
    
    'drift_D': 0.0,
    
    # S=1，所以 sigD 是 1x2 的矩阵
    'sigD': torch.tensor([
        [0.24/np.sqrt(2), 0.24/np.sqrt(2)]
    ], dtype=torch.float32),
    
    'muE': torch.zeros(I),
    
    # 两个代理人，所以 sigE 是 2x2 的矩阵
    'sigE': torch.tensor([
        [0.0, 0.6/np.sqrt(2)],  
        [0.0, 0.4/np.sqrt(2)]   
    ], dtype=torch.float32),
    
    'alpha': torch.tensor([0.4, 0.6]),
    'epsilon': 1e-6,
    'N_option': 1.0, 
    'a_expo': 0.5,   
    'x2': 0.0,
    'checkpoint_path': 'radner_single_asset_put.pth',
    
    't_train': t_train, 'W_train': W_train,
    't_valid': t_valid, 'W_valid': W_valid
}

# ！！！直接使用你写的 Multi Solver！！！
# solver_single = RadnerEquilibriumSolver_multi_put(config_single)
# solver_single.train(NIter, epoch, patience)


class RadnerEquilibriumSolver_Linear(RadnerEquilibriumSolver_multi_put):
    def endowment_process(self, t, X, i):
        """
        极简线性禀赋，直接对应 sigE[i] * X
        这样可以确保你的 S_exact, Y_exact 数学公式完美成立
        """
        x2 = self.x2
        X_shifted = X + x2
        # 直接返回内积，不加额外正负号
        return (self.sigE[i, :] * X_shifted).sum(dim=-1, keepdim=True)

# ---------------------------------------------------------------------------
# 2. 重新初始化并训练
# ---------------------------------------------------------------------------
print("\n=== Training Linear Case for Analytical Verification ===")
solver_linear = RadnerEquilibriumSolver_Linear(config_single)
solver_linear.train(NIter=1000, epoch=5, patience=2)  # 线性系统极其容易收敛，5个 epoch 足够了

# ---------------------------------------------------------------------------
# 3. 生成测试路径并进行预测
# ---------------------------------------------------------------------------
print("\n=== Generating Predictions vs Exact Solutions ===")
M_test = 5 # 画 5 条轨迹进行对比即可
t_test, W_test = BrownianMotionGenerator.generate(M_test, N, D, T)
t_test = t_test.to(solver_linear.device)
W_test = W_test.to(solver_linear.device)

# 神经网络预测
with torch.no_grad():
    Y_pred, Z_pred, Th_pred = solver_linear.predict(t_test, W_test)

# 提取预测值
S_pred = Y_pred[0]       # 股票价格 (M_test, N+1, 1)
R1_pred = Y_pred[1]      # Agent 1 效用 (M_test, N+1, 1)
Theta1_pred = Th_pred[0] # Agent 1 策略 (M_test, N+1, S)

# 计算解析解 (Exact)
S_true = solver_linear.S_exact(t_test, W_test)
R1_true = solver_linear.Y_exact(t_test, W_test, 0)
theta_true = solver_linear.get_analytical_theta()

# ---------------------------------------------------------------------------
# 4. 可视化检验 (Plotting)
# ---------------------------------------------------------------------------
t_np = t_test[0, :, 0].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图 1: 股票价格 S(t)
for k in range(M_test):
    axes[0].plot(t_np, S_pred[k, :, 0].cpu().numpy(), color='blue', alpha=0.6, 
                 label='NN Predict' if k==0 else "")
    axes[0].plot(t_np, S_true[k, :, 0].cpu().numpy(), color='red', linestyle='--', 
                 label='Exact Analytic' if k==0 else "")
axes[0].set_title("Stock Price $S_t$")
axes[0].set_xlabel("Time t")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图 2: 代理人 1 的财富效用 R^1(t)
for k in range(M_test):
    axes[1].plot(t_np, R1_pred[k, :, 0].cpu().numpy(), color='green', alpha=0.6, 
                 label='NN Predict' if k==0 else "")
    axes[1].plot(t_np, R1_true[k, :, 0].cpu().numpy(), color='orange', linestyle='--', 
                 label='Exact Analytic' if k==0 else "")
axes[1].set_title("Agent 1 Utility $R^1_t$")
axes[1].set_xlabel("Time t")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图 3: 代理人 1 的最优头寸 Theta^1(t)
for k in range(M_test):
    axes[2].plot(t_np, Theta1_pred[k, :, 0].cpu().numpy(), color='purple', alpha=0.4,
                 label='NN Predict Path' if k==0 else "")
axes[2].axhline(y=theta_true[0].item(), color='red', linestyle='--', linewidth=2, 
                label=f'Exact $\\theta^1$ = {theta_true[0].item():.4f}')
axes[2].set_title("Agent 1 Strategy $\\theta^1_t$")
axes[2].set_xlabel("Time t")
axes[2].set_ylim(theta_true[0].item() - 0.5, theta_true[0].item() + 0.5)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()