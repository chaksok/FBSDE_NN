import torch
import numpy as np
import copy
import torch.nn.functional as F
from core.data_generator import BrownianMotionGenerator
from models.radner_terminal_muti_solver import RadnerEquilibriumSolverMulti4
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Training hyperparameters & Dimensions (Multi-Asset)
# ---------------------------------------------------------------------------
epoch = 8        # Number of training epochs (Linear case converges fast)
NIter = 1000    # Number of iterations per epoch
patience = 2     # Patience for early stopping

T = 1.0
M = 128          # Batch size
N = 100          # Time steps
K = 2000         # Validation size

# 多资产核心维度设定
S = 2            # Number of stocks (M in the paper)
I = 2            # Number of agents
D = 3            # Number of Brownian motions (W1: Stock A, W2: Stock B, W3: Weather)

print("Generating training data...")
t_train, W_train = BrownianMotionGenerator.generate(M * NIter, N, D, T)
t_valid, W_valid = BrownianMotionGenerator.generate(K, N, D, T)

# ---------------------------------------------------------------------------
# 2. Configuration Dictionary
# ---------------------------------------------------------------------------
config_multi = {
    'T': T, 'M': M, 'N': N, 'D': D, 
    'S': S, 'I': I, 'K': K,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 网络架构自适应维度：Y输出 S+I 维， Z输出 (S+I)*D 维
    'layers_y': [D + 1] + 5 * [256] + [S + I],
    'layers_z': [D + 1] + 5 * [256] + [(S + I) * D],
    
    'drift_D': 0.0,
    
    # 股票波动率 (S=2, D=3): 股票A暴露于W1，股票B暴露于W2
    'sigD': torch.tensor([
        [0.2, 0.1, 0.0],  # Stock A
        [0.1, 0.05, 0.15]   # Stock B
    ], dtype=torch.float32),
    
    'muE': torch.zeros(I),
    
    # 禀赋波动率 (I=2, D=3): 两个代理人只暴露于天气风险 W3
    'sigE': torch.tensor([
        [0.1, 0.3, 0.05],  # Agent 1
        [0.05, 0.15, 0.2]   # Agent 2
    ], dtype=torch.float32),
    
    'alpha': torch.tensor([0.4, 0.6]),
    'epsilon': 1e-6,
    'N_option': 1.0, 
    'a_expo': 0.5,   
    'x2': 0.0,
    'checkpoint_path': 'radner_multi_asset_linear.pth',
    
    't_train': t_train, 'W_train': W_train,
    't_valid': t_valid, 'W_valid': W_valid
}

# ---------------------------------------------------------------------------
# 3. Custom Solver Class for Multi-Stock (Linear Analytical Solutions)
# ---------------------------------------------------------------------------
class RadnerEquilibriumSolver_Linear(RadnerEquilibriumSolverMulti4):
    def __init__(self, config):
        super().__init__(config)
        self.x2 = config['x2']
        
    def endowment_process(self, t, X, i):
        """
        极简线性禀赋，直接对应 sigE[i] * X
        这样可以确保 S_exact, Y_exact 的解析公式完美成立
        """
        x2 = self.x2
        X_shifted = X + x2
        # 直接返回内积
        return (self.sigE[i, :] * X_shifted).sum(dim=-1, keepdim=True)
    
    def S_exact(self, t, X):       
        """
        Compute the exact solution S(t,W) of the Radner equilibrium FBSDE for multiple assets.
        Returns shape: (Batch, N, S)
        """
        term1 = self.sigD.sum(dim=0)                                # (D,)
        term2 = (self.alpha.unsqueeze(-1) * self.sigE).sum(dim=0)   # (D,)
        drift_factor = term1 + term2                                # (D,)
        
        mu_S = torch.matmul(self.sigD, drift_factor)                # (S,)
        diffusion = torch.matmul(X, self.sigD.T)                    # (Batch, N, S)
        
        return (t - 1) * mu_S + diffusion
    
    def Y_exact(self, t, X, i):
        """
        Compute the exact solution Y(t,W) for agent i.
        Returns shape: (Batch, N, 1)
        """
        term1 = self.sigD.sum(dim=0)  
        term2 = (self.alpha.unsqueeze(-1) * self.sigE).sum(dim=0)  
        diff_i = term1 + term2 - self.sigE[i]  
        
        # 矩阵求逆计算投影
        aaT = torch.matmul(self.sigD, self.sigD.T)                  # (S, S)
        inv_aaT = torch.linalg.inv(aaT)                             # (S, S)
        
        diff_i_aT = torch.matmul(diff_i, self.sigD.T)               # (S,)
        proj_coef = torch.matmul(diff_i_aT, inv_aaT)                # (S,)
        proj_vec = torch.matmul(proj_coef, self.sigD)               # (D,)
        
        part1 = 0.5 * torch.sum(proj_vec**2)                  
        part2 = 0.5 * torch.sum(self.sigE[i]**2)              
        c_i = part2 - part1                                         # 标量
        
        diffusion = torch.matmul(X, self.sigE[i])                   # (Batch, N)
        return (t - 1) * c_i + diffusion.unsqueeze(-1)
    
    def get_analytical_theta(self):
        """
        Analytical Radner equilibrium strategies for multiple assets.
        Returns: Tensor of shape (I, S)
        """
        sum_b = (self.alpha.unsqueeze(-1) * self.sigE).sum(dim=0)   # (D,)
        diff = sum_b.unsqueeze(0) - self.sigE                       # (I, D)
        
        aaT = torch.matmul(self.sigD, self.sigD.T)                  # (S, S)
        inv_aaT = torch.linalg.inv(aaT)                             # (S, S)
        
        hedge_demand = torch.matmul(torch.matmul(diff, self.sigD.T), inv_aaT) # (I, S)
        theta = 1.0 + hedge_demand                                  # 自动广播
        
        return theta  # (I, S)

# ---------------------------------------------------------------------------
# 4. Initialization and Training execution
# ---------------------------------------------------------------------------
print("\n=== Training Linear Case for Multi-Asset Verification ===")
solver_multi = RadnerEquilibriumSolver_Linear(config_multi)
solver_multi.train(NIter=NIter, epoch=epoch, patience=patience)

# # ---------------------------------------------------------------------------
# # 5. Generate Test Paths and Evaluate
# # ---------------------------------------------------------------------------
# print("\n=== Generating Predictions vs Exact Solutions ===")
# M_test = 5 
# t_test, W_test = BrownianMotionGenerator.generate(M_test, N, D, T)
# t_test = t_test.to(solver_multi.device)
# W_test = W_test.to(solver_multi.device)

# with torch.no_grad():
#     Y_pred, Z_pred, Th_pred = solver_multi.predict(t_test, W_test)

# # 多资产输出提取: Y_pred 包含 S+I 个输出列表
# # 前 S 个是股票价格，后 I 个是代理人的延续价值 (Utility)
# S1_pred = Y_pred[0]      # Stock A (M_test, N+1, 1)
# S2_pred = Y_pred[1]      # Stock B (M_test, N+1, 1)
# R1_pred = Y_pred[S]      # Agent 1 Utility (M_test, N+1, 1)

# # 策略提取: Th_pred 包含 I 个输出，每个形状为 (M_test, N+1, S)
# Theta1_pred = Th_pred[0] # Agent 1 Strategies over all S stocks

# # 解析解提取
# S_true = solver_multi.S_exact(t_test, W_test)      # (M_test, N+1, S)
# R1_true = solver_multi.Y_exact(t_test, W_test, 0)  # (M_test, N+1, 1)
# theta_true = solver_multi.get_analytical_theta()   # (I, S)

# # ---------------------------------------------------------------------------
# # 6. 可视化检验 (Plotting 2x2 Grid)
# # ---------------------------------------------------------------------------
# t_np = t_test[0, :, 0].cpu().numpy()

# fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# # [0, 0] Stock A Price S_1(t)
# for k in range(M_test):
#     axes[0, 0].plot(t_np, S1_pred[k, :, 0].cpu().numpy(), color='blue', alpha=0.6, 
#                     label='NN Predict' if k==0 else "")
#     axes[0, 0].plot(t_np, S_true[k, :, 0].cpu().numpy(), color='red', linestyle='--', 
#                     label='Exact Analytic' if k==0 else "")
# axes[0, 0].set_title("Stock A Price $S_{1,t}$")
# axes[0, 0].set_xlabel("Time t")
# axes[0, 0].legend()
# axes[0, 0].grid(True, alpha=0.3)

# # [0, 1] Stock B Price S_2(t)
# for k in range(M_test):
#     axes[0, 1].plot(t_np, S2_pred[k, :, 0].cpu().numpy(), color='green', alpha=0.6, 
#                     label='NN Predict' if k==0 else "")
#     axes[0, 1].plot(t_np, S_true[k, :, 1].cpu().numpy(), color='red', linestyle='--', 
#                     label='Exact Analytic' if k==0 else "")
# axes[0, 1].set_title("Stock B Price $S_{2,t}$")
# axes[0, 1].set_xlabel("Time t")
# axes[0, 1].legend()
# axes[0, 1].grid(True, alpha=0.3)

# # [1, 0] Agent 1 Strategy on Stock A
# for k in range(M_test):
#     axes[1, 0].plot(t_np, Theta1_pred[k, :, 0].cpu().numpy(), color='purple', alpha=0.4,
#                     label='NN Predict Path' if k==0 else "")
# theta_true_A = theta_true[0, 0].item()
# axes[1, 0].axhline(y=theta_true_A, color='red', linestyle='--', linewidth=2, 
#                    label=f'Exact $\\theta^{{1,A}}$ = {theta_true_A:.4f}')
# axes[1, 0].set_title("Agent 1 Strategy on Stock A ($\\theta^{1,A}_t$)")
# axes[1, 0].set_xlabel("Time t")
# axes[1, 0].set_ylim(theta_true_A - 0.5, theta_true_A + 0.5)
# axes[1, 0].legend()
# axes[1, 0].grid(True, alpha=0.3)

# # [1, 1] Agent 1 Strategy on Stock B
# for k in range(M_test):
#     axes[1, 1].plot(t_np, Theta1_pred[k, :, 1].cpu().numpy(), color='orange', alpha=0.4,
#                     label='NN Predict Path' if k==0 else "")
# theta_true_B = theta_true[0, 1].item()
# axes[1, 1].axhline(y=theta_true_B, color='red', linestyle='--', linewidth=2, 
#                    label=f'Exact $\\theta^{{1,B}}$ = {theta_true_B:.4f}')
# axes[1, 1].set_title("Agent 1 Strategy on Stock B ($\\theta^{1,B}_t$)")
# axes[1, 1].set_xlabel("Time t")
# axes[1, 1].set_ylim(theta_true_B - 0.5, theta_true_B + 0.5)
# axes[1, 1].legend()
# axes[1, 1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt

# ... [Keep your previous configuration and solver class definitions exactly as they are] ...

# ---------------------------------------------------------------------------
# 5. Generate Test Paths and Evaluate
# ---------------------------------------------------------------------------
print("\n=== Generating Predictions vs Exact Solutions ===")
M_test = 200 # Using a larger test batch for stable metric calculation
t_test, W_test = BrownianMotionGenerator.generate(M_test, N, D, T)
t_test = t_test.to(solver_multi.device)
W_test = W_test.to(solver_multi.device)

with torch.no_grad():
    Y_pred, Z_pred, Th_pred = solver_multi.predict(t_test, W_test)

# S_pred contains S elements, each of shape (M_test, N+1, 1)
S_pred_list = Y_pred[:S]
# R_pred contains I elements, each of shape (M_test, N+1, 1)
R_pred_list = Y_pred[S:]

# Exact Solutions
S_true = solver_multi.S_exact(t_test, W_test)         # (M_test, N+1, S)
theta_true = solver_multi.get_analytical_theta()      # (I, S)
sigD_true = solver_multi.sigD                         # (S, D)

# ---------------------------------------------------------------------------
# 6. Calculate Validation Metrics (Table Reproduction)
# ---------------------------------------------------------------------------
# 6.1 Initial Stock Price Error
S0_pred = torch.cat([S_pred_list[s][0, 0, :] for s in range(S)]) # (S,)
S0_true = S_true[0, 0, :]                                        # (S,)
S0_abs_err = torch.abs(S0_pred - S0_true).max().item()
S0_rel_err = (S0_abs_err / torch.abs(S0_true).max().item())

# 6.2 Initial Certainty Equivalents Error
Y0_err_list = []
for i in range(I):
    Y0_pred_i = R_pred_list[i][0, 0, 0].item()
    Y0_true_i = solver_multi.Y_exact(t_test, W_test, i)[0, 0, 0].item()
    Y0_err_list.append(abs(Y0_pred_i - Y0_true_i))
max_Y0_err = max(Y0_err_list)

# 6.3 Stock Volatility Error
# Z_pred shape is (M_test, N, (S+I)*D). The first S*D columns are the stock volatilities.
# Z_pred is a list of (S+I) tensors, each of shape (M_test, N, D).
# We take the first S elements (stock volatilities) and stack them along dim=2
# Z_pred[s] shape: (M_test, N+1, 1, D) → squeeze掉多余的dim=2 → (M_test, N+1, D)
sigma_pred = torch.stack([Z_pred[s].squeeze(2) for s in range(S)], dim=2)  # (M_test, N+1, S, D)

sigma_true = sigD_true.unsqueeze(0).unsqueeze(0).expand(M_test, N+1, S, D)  # N → N+1

sig_abs_err   = torch.norm(sigma_pred - sigma_true, p=2, dim=-1).mean().item()
sig_true_norm = torch.norm(sigma_true, p=2, dim=-1).mean().item()
sig_rel_err   = sig_abs_err / sig_true_norm

# 6.4 Portfolio Strategies Max RRMSE
rrmse_list = []
for i in range(I):
    # Drop the last time step to match Z/control dimensions if necessary, or compute over all available
    # Th_pred[i] shape: (M_test, N+1, S) -> We evaluate over the first N steps
    th_pred_i = Th_pred[i][:, :-1, :] 
    th_true_i = theta_true[i].unsqueeze(0).unsqueeze(0).expand(M_test, N, S)
    
    mse = torch.mean((th_pred_i - th_true_i)**2)
    true_pow = torch.mean(th_true_i**2)
    rrmse = torch.sqrt(mse) / torch.sqrt(true_pow)
    rrmse_list.append(rrmse.item())
max_rrmse = max(rrmse_list)

# 6.5 Nondegeneracy (D_epsilon)
epsilon = 1e-4
# Norm of volatility for each stock m at each time step
sigma_pred_norm = torch.norm(sigma_pred, p=2, dim=-1) # (M_test, N, S)
D_eps = torch.mean((sigma_pred_norm < epsilon).float()).item()

print("\n" + "="*65)
print(f"{'Quantity':<30} | {'Diagnostic':<15} | {'Value'}")
print("-" * 65)
print(f"{'Initial stock price':<30} | |S0_hat - S0|   | {S0_abs_err:.2e} ({S0_rel_err:.2e})")
print(f"{'Initial certainty equivalents':<30} | max |Y0_hat-Y0| | {max_Y0_err:.2e}")
print(f"{'Stock volatility':<30} | ||sig_hat - b0||| {sig_abs_err:.2e} ({sig_rel_err:.2e})")
print(f"{'Portfolio strategies':<30} | max RRMSE       | {max_rrmse:.2e}")
print(f"{'Nondegeneracy':<30} | D_{epsilon}          | {D_eps:.2f}")
print("=" * 65 + "\n")

# ---------------------------------------------------------------------------
# 7. Visualization (4x2 Grid)
# ---------------------------------------------------------------------------
t_np = t_test[0, :, 0].cpu().numpy()
num_paths_to_plot = 5 

# 将画布扩大为 4 行 2 列，调整高度以容纳新图表
fig, axes = plt.subplots(4, 2, figsize=(14, 20))

# --- Row 1: Stock Prices ---
for k in range(num_paths_to_plot):
    axes[0, 0].plot(t_np, S_pred_list[0][k, :, 0].cpu().numpy(), color='blue', alpha=0.6, label='NN Predict' if k==0 else "")
    axes[0, 0].plot(t_np, S_true[k, :, 0].cpu().numpy(), color='red', linestyle='--', label='Exact' if k==0 else "")
axes[0, 0].set_title("Stock A Price ($S_{1,t}$)")
axes[0, 0].set_xlabel("Time t")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

for k in range(num_paths_to_plot):
    axes[0, 1].plot(t_np, S_pred_list[1][k, :, 0].cpu().numpy(), color='green', alpha=0.6, label='NN Predict' if k==0 else "")
    axes[0, 1].plot(t_np, S_true[k, :, 1].cpu().numpy(), color='red', linestyle='--', label='Exact' if k==0 else "")
axes[0, 1].set_title("Stock B Price ($S_{2,t}$)")
axes[0, 1].set_xlabel("Time t")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# --- Row 2: Agent Utilities (Certainty Equivalents) ---
for i in range(I):
    R_true_i = solver_multi.Y_exact(t_test, W_test, i)
    for k in range(num_paths_to_plot):
        axes[1, i].plot(t_np, R_pred_list[i][k, :, 0].cpu().numpy(), color='purple', alpha=0.6, label='NN Predict' if k==0 else "")
        axes[1, i].plot(t_np, R_true_i[k, :, 0].cpu().numpy(), color='orange', linestyle='--', label='Exact' if k==0 else "")
    axes[1, i].set_title(f"Agent {i+1} Utility ($R^{i+1}_t$)")
    axes[1, i].set_xlabel("Time t")
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

# --- Row 3: Agent 1 Strategies (Stock A & Stock B) ---
for k in range(num_paths_to_plot):
    axes[2, 0].plot(t_np, Th_pred[0][k, :, 0].cpu().numpy(), color='teal', alpha=0.4, label='NN Predict Path' if k==0 else "")
th_true_1A = theta_true[0, 0].item()
axes[2, 0].axhline(y=th_true_1A, color='red', linestyle='--', linewidth=2, label=f'Exact = {th_true_1A:.4f}')
axes[2, 0].set_title("Agent 1 Strategy on Stock A ($\\theta^{1,A}_t$)")
axes[2, 0].set_xlabel("Time t")
axes[2, 0].set_ylim(th_true_1A - 0.5, th_true_1A + 0.5)
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

for k in range(num_paths_to_plot):
    axes[2, 1].plot(t_np, Th_pred[0][k, :, 1].cpu().numpy(), color='teal', alpha=0.4, linestyle='-.', label='NN Predict Path' if k==0 else "")
th_true_1B = theta_true[0, 1].item()
axes[2, 1].axhline(y=th_true_1B, color='red', linestyle='--', linewidth=2, label=f'Exact = {th_true_1B:.4f}')
axes[2, 1].set_title("Agent 1 Strategy on Stock B ($\\theta^{1,B}_t$)")
axes[2, 1].set_xlabel("Time t")
axes[2, 1].set_ylim(th_true_1B - 0.5, th_true_1B + 0.5)
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# --- Row 4: Agent 2 Strategies (Stock A & Stock B) ---
for k in range(num_paths_to_plot):
    axes[3, 0].plot(t_np, Th_pred[1][k, :, 0].cpu().numpy(), color='brown', alpha=0.4, label='NN Predict Path' if k==0 else "")
th_true_2A = theta_true[1, 0].item()
axes[3, 0].axhline(y=th_true_2A, color='red', linestyle='--', linewidth=2, label=f'Exact = {th_true_2A:.4f}')
axes[3, 0].set_title("Agent 2 Strategy on Stock A ($\\theta^{2,A}_t$)")
axes[3, 0].set_xlabel("Time t")
axes[3, 0].set_ylim(th_true_2A - 0.5, th_true_2A + 0.5)
axes[3, 0].legend()
axes[3, 0].grid(True, alpha=0.3)

for k in range(num_paths_to_plot):
    axes[3, 1].plot(t_np, Th_pred[1][k, :, 1].cpu().numpy(), color='brown', alpha=0.4, linestyle='-.', label='NN Predict Path' if k==0 else "")
th_true_2B = theta_true[1, 1].item()
axes[3, 1].axhline(y=th_true_2B, color='red', linestyle='--', linewidth=2, label=f'Exact = {th_true_2B:.4f}')
axes[3, 1].set_title("Agent 2 Strategy on Stock B ($\\theta^{2,B}_t$)")
axes[3, 1].set_xlabel("Time t")
axes[3, 1].set_ylim(th_true_2B - 0.5, th_true_2B + 0.5)
axes[3, 1].legend()
axes[3, 1].grid(True, alpha=0.3)

plt.tight_layout()

# 保存图表为高分辨率 PNG 文件
# dpi=300 保证了图片的清晰度，bbox_inches='tight' 可以防止边缘被裁剪
save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/multi_asset_equilibrium_results.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n图表已成功保存至: {save_path}")

plt.show() # 如果你只想保存不想在运行中弹窗，可以把这行注释掉