import torch
import numpy as np
import matplotlib.pyplot as plt
from core.data_generator import BrownianMotionGenerator
from models.radner_terminal_muti_solver import RadnerEquilibriumSolverMulti4
from utils.multi_visualization import MultiAssetVisualizer

# 假设你已经导入了之前的模块，例如：
# from core.data_generator import BrownianMotionGenerator
# from your_module import RadnerEquilibriumSolverMulti4, MultiAssetVisualizer

# =====================================================================
# 1. 经济学场景与参数配置 (The Economic Scenario Setup)
# =====================================================================
T = 1.0
M = 128         # Batch size (训练轨迹数)
N = 50       # 时间步数
K = 2000         # 验证集轨迹数

S = 2  # 两支股票: A 和 B
I = 2  # 两个代理人: Agent 1, Agent 2
D = 3  # 三维风险源: W1, W2, W3 (天气)

# 生成训练和验证数据
print("Generating Brownian Motion paths...")
t_train, W_train = BrownianMotionGenerator.generate(M * 1000, N, D, T) # NIter=1000
t_valid, W_valid = BrownianMotionGenerator.generate(K, N, D, T)

config_app = {
    'T': T, 'M': M, 'N': N, 'D': D, 'S': S, 'I': I, 'K': K,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # 网络架构: Y输出4维(S+I), Z输出12维((S+I)*D)
    'layers_y': [D + 1] + 5 * [256] + [S + I],
    'layers_z': [D + 1] + 5 * [256] + [(S + I) * D],
    
    'drift_D': 0.0,
    
    # 股票波动率矩阵 (2x3): A只暴露于W1, B只暴露于W2, 对W3均无敞口
    'sigD': torch.tensor([
        [0.2/np.sqrt(2), 0.0, 0.0],  # Stock A (科技)
        [0.0, 0.2/np.sqrt(2), 0.0]   # Stock B (农业)
    ], dtype=torch.float32),
    
    'muE': torch.zeros(I),
    
    # 禀赋波动率矩阵 (2x3): 代理人只对天气风险 W3 有敞口
    'sigE': torch.tensor([
        [0.0, 0.0,  0.5/np.sqrt(2)],  # Agent 1 暴露于天气
        [0.0, 0.0, -0.5/np.sqrt(2)]   # Agent 2 做对手盘
    ], dtype=torch.float32),
    
    'alpha': torch.tensor([0.4, 0.6]), # 风险厌恶系数
    'epsilon': 1e-6,
    'N_option': 1.0, 
    'a_expo': 0.8,   # a=0.8 表示强烈的非线性期权特征 (制造巨大的 Gamma 风险)
    'x2': 0.0,
    'checkpoint_path': 'multi_asset_application.pth',
    
    't_train': t_train, 'W_train': W_train,
    't_valid': t_valid, 'W_valid': W_valid
}

# =====================================================================
# 2. 定义包含非线性期权的求解器
# =====================================================================
class RadnerEquilibriumSolver_App(RadnerEquilibriumSolverMulti4):
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
        self.x2 = config['x2']
        
    def endowment_process(self, t, X, i):
        """代理人持有基于天气 W3 的非线性期权"""
        a = self.a_expo
        x2 = self.x2
        X_shifted = X + x2
        
        # 提取天气敞口 (内积自动定位到 W3)
        X_weather = (self.sigE[i, :] * X_shifted).sum(dim=-1, keepdim=True)
        
        # 非线性期权收益: (1-a)*线性 + a*看跌期权
        payoff = (1 - a) * X_weather + a * torch.minimum(X_weather, torch.zeros_like(X_weather))

        if i == 0:
            return -self.N_option * payoff
        else:
            return self.N_option * payoff

# =====================================================================
# 3. 训练模型
# =====================================================================
print("\n=== Initializing Multi-Asset Solver ===")
solver_app = RadnerEquilibriumSolver_App(config_app)

print("\n=== Starting Training Phase ===")
# 提示: 为了快速看到结果，这里 epoch 设为 3。你的毕业论文最终出图建议设为 10-20。
solver_app.train(NIter=1000, epoch=12, patience=2)

# =====================================================================
# 4. 生成测试数据并预测 (Testing and Prediction)
# =====================================================================
print("\n=== Generating Test Paths and Predicting ===")
M_test = 5000  # 测试集要足够大，画出来的 3D 曲面才会平滑
t_test, W_test = BrownianMotionGenerator.generate(M_test, N, D, T)
t_test = t_test.to(solver_app.device)
W_test = W_test.to(solver_app.device)

with torch.no_grad():
    Y_path, Z_path, Th_path = solver_app.predict(t_test, W_test)

# =====================================================================
# 5. 调用可视化模块出图 (Visualization)
# =====================================================================
print("\n=== Rendering 3D Economic Phenomenon Plots ===")
# 这里调用的是我们上一条对话中写的 MultiAssetVisualizer 类
visualizer = MultiAssetVisualizer(t_test, W_test, Y_path, Z_path, Th_path, solver_app)

# 图 1: 交叉风险溢价 (Cross-Equity Premium)
print("Plotting Cross-Equity Premium...")
visualizer.plot_cross_equity_premium()

# 图 2: 行列式深渊 (Determinant Dimple - 证明非退化)
print("Plotting Determinant Dimple...")
visualizer.plot_determinant_dimple()

# 图 3: 内生相关性 (Endogenous Correlation)
print("Plotting Endogenous Correlation...")
visualizer.plot_endogenous_correlation()

# 图 4: 动态交叉对冲轨迹 (Flight to Quality)
print("Plotting Flight to Quality Behavior...")
visualizer.plot_flight_to_quality()