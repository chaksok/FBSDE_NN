import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MultiAssetVisualizer:
    def __init__(self, t_test, W_test, Y_path, Z_path, Th_path, solver):
        """
        初始化可视化工具
        """
        self.t = t_test[:, :, 0].cpu().numpy()
        self.W = W_test.cpu().numpy()
        
        # 获取精确的形状参数
        M_test, N_plus_1 = self.t.shape
        self.S = solver.S
        self.D = solver.D
        self.alpha = solver.alpha.cpu().numpy()
        
        # ！！！终极修复：强制 Reshape，彻底消除任何多余的空维度 (例如 unsqueeze 带来的 1)！！！
        self.sigma_A = Z_path[0].cpu().numpy().reshape(M_test, N_plus_1, self.D)
        self.sigma_B = Z_path[1].cpu().numpy().reshape(M_test, N_plus_1, self.D)
        
        self.gamma_1 = Z_path[2].cpu().numpy().reshape(M_test, N_plus_1, self.D)
        self.gamma_2 = Z_path[3].cpu().numpy().reshape(M_test, N_plus_1, self.D)
        
        # 同样清洗 Theta 头寸的形状为 (M, N+1, S)
        self.theta_1 = Th_path[0].cpu().numpy().reshape(M_test, N_plus_1, self.S)
        self.theta_2 = Th_path[1].cpu().numpy().reshape(M_test, N_plus_1, self.S)
        
        # 降采样设置 (为了 3D 画图不卡顿)
        self.step_m = 20  # 每 20 条轨迹取 1 条
        self.step_n = 2   # 每 2 个时间步取 1 个
        
    def _get_grid(self):
        # 提取天气风险 W3 (索引为 2) 作为 X 轴
        W3 = self.W[::self.step_m, ::self.step_n, 2] * (0.2 / np.sqrt(2))
        T = self.t[::self.step_m, ::self.step_n]
        return W3.flatten(), T.flatten()

    def plot_determinant_dimple(self):
        """
        方向 4: 数学意义 - 行列式深渊 (证明协方差矩阵非退化)
        """
        W3_flat, T_flat = self._get_grid()
        
        sigA = self.sigma_A[::self.step_m, ::self.step_n, :]
        sigB = self.sigma_B[::self.step_m, ::self.step_n, :]
        
        # 计算每个点的协方差矩阵 sigma * sigma^T
        # sigma 形状为 (M, N, 2, 3)
        sigma_matrix = np.stack([sigA, sigB], axis=2) 
        sigma_T = np.transpose(sigma_matrix, axes=(0, 1, 3, 2))
        cov_matrix = np.matmul(sigma_matrix, sigma_T) # (M, N, 2, 2)
        
        # 计算 2x2 矩阵的行列式: ad - bc
        det = cov_matrix[:,:,0,0]*cov_matrix[:,:,1,1] - cov_matrix[:,:,0,1]*cov_matrix[:,:,1,0]
        det_flat = np.abs(det.flatten())
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(W3_flat, T_flat, det_flat, cmap='viridis', edgecolor='none', alpha=0.9)
        
        ax.set_title("The Determinant Dimple (Proof of Non-degeneracy)")
        ax.set_xlabel("Weather State (Unspanned Risk)")
        ax.set_ylabel("Time")
        ax.set_zlabel("Det(Covariance Matrix)")
        ax.view_init(elev=20, azim=-45)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def plot_cross_equity_premium(self):
        """
        方向 1: 经济学意义 - 交叉风险溢价
        股票 A (科技股) 的溢价如何被天气风险 (W3) 扭曲
        """
        W3_flat, T_flat = self._get_grid()
        
        sigA = self.sigma_A[::self.step_m, ::self.step_n, :]
        sigB = self.sigma_B[::self.step_m, ::self.step_n, :]
        g1 = self.gamma_1[::self.step_m, ::self.step_n, :]
        g2 = self.gamma_2[::self.step_m, ::self.step_n, :]
        
        # 市场总对冲需求
        bar_gamma = self.alpha[0] * g1 + self.alpha[1] * g2
        
        # 股票 A 的风险溢价: (bar_gamma + sigA + sigB) dot sigA
        mu_A = np.sum((bar_gamma + sigA + sigB) * sigA, axis=2)
        mu_A_flat = mu_A.flatten()
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(W3_flat, T_flat, mu_A_flat, cmap='magma', edgecolor='none', alpha=0.9)
        
        ax.set_title("Cross-Equity Premium on Stock A Driven by Weather Risk")
        ax.set_xlabel("Weather State (W3)")
        ax.set_ylabel("Time")
        ax.set_zlabel("Risk Premium of Stock A")
        ax.view_init(elev=25, azim=-60)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def plot_endogenous_correlation(self):
        """
        方向 2: 经济学意义 - 内生相关性
        基本面正交的股票 A 和 B，因为代理人的交叉对冲产生市场定价的相关性
        """
        W3_flat, T_flat = self._get_grid()
        
        sigA = self.sigma_A[::self.step_m, ::self.step_n, :]
        sigB = self.sigma_B[::self.step_m, ::self.step_n, :]
        
        # Cov(A, B) = sigA dot sigB
        cov_AB = np.sum(sigA * sigB, axis=2)
        cov_AB_flat = cov_AB.flatten()
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(W3_flat, T_flat, cov_AB_flat, cmap='coolwarm', edgecolor='none', alpha=0.9)
        
        ax.set_title("Endogenous Covariance between Stock A and B")
        ax.set_xlabel("Weather State (W3)")
        ax.set_ylabel("Time")
        ax.set_zlabel("Covariance(A, B)")
        ax.view_init(elev=20, azim=120)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    def plot_flight_to_quality(self):
        """
        方向 3: 经济学意义 - 动态交叉对冲轨迹
        展示临近到期时 (t=0.9)，代理人 1 的投资组合权重在状态空间中的散点分布
        """
        # 取 t=0.90 的截面
        idx_t = int(self.theta_1.shape[1] * 0.90)
        
        theta_A = self.theta_1[:, idx_t, 0]
        theta_B = self.theta_1[:, idx_t, 1]
        W3_state = self.W[:, idx_t, 2] * (0.2 / np.sqrt(2))
        
        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(theta_A, theta_B, c=W3_state, cmap='RdYlBu', alpha=0.6, s=15)
        plt.colorbar(scatter, label="Weather State (Bad Weather < 0 < Good Weather)")
        
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f"Agent 1 Portfolio Weights (Flight to Quality) at t={self.t[0, idx_t]:.2f}")
        plt.xlabel("Position in Stock A")
        plt.ylabel("Position in Stock B")
        plt.grid(True, alpha=0.3)
        plt.show()

# ==========================================
# 运行可视化代码的示例
# ==========================================
# 假设你已经运行了 solver_multi.predict(t_test, W_test) 并得到了 Y, Z, Th
# visualizer = MultiAssetVisualizer(t_test, W_test, Y_path, Z_path, Th_path, solver)
# visualizer.plot_determinant_dimple()
# visualizer.plot_cross_equity_premium()
# visualizer.plot_endogenous_correlation()
# visualizer.plot_flight_to_quality()