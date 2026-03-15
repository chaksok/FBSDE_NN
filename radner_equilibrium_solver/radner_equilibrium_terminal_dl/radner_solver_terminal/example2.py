"""
Example script for solving Radner equilibria using FBSDE neural networks.

This script demonstrates how to configure, train, and evaluate a neural network solver
for Radner equilibrium problems in incomplete markets. It uses the RadnerNeuralNetwork1
class to approximate the solution to the associated forward-backward stochastic
differential equations.

The example includes:
- Problem setup with market parameters
- Neural network training with early stopping
- Prediction and visualization of results
- Error analysis and plotting

Usage:
    python example.py

Requirements:
    - PyTorch
    - NumPy
    - Matplotlib
    - Seaborn
    - Plotly
    - Custom modules: models.radner_terminal_BSDE_solver, utils.visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import time
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import plotly.express as px
sns.set()
# plt.style.use('ggplot')

# Import custom modules
from models.radner_terminal_BSDE_solver import RadnerNeuralNetwork1
from utils.visualization import RadnerVisualizer
from core.data_generator import BrownianMotionGenerator

# Set random seeds for reproducibility
torch.manual_seed(1)
np.random.seed(1)

    
print("="*60)
print("Radner Equilibrium FBSDE Solver")
print("="*60)

# =============================================================================
# Problem Configuration
# =============================================================================

# Time horizon and discretization
T = 1  # Terminal time
M = 100  # Number of trajectories per batch
N = 100  # Number of time steps
D = 4  # Dimension of Brownian motion (state space)
I = 3  # Number of agents
K = 1000  # Number of validation trajectories

# Market parameters for Radner equilibrium
drift_D = 0.0  # Drift coefficient for dividend process
sigD = torch.tensor([[0.2, 0.0, 0., 0.0]])  # Volatility matrix for dividend
sigE = torch.tensor([
            [0.18, 0.24, 0.0, 0.0],
            [0.18, 0.0, 0.24, 0.0],
            [0.18, 0.0, 0.0, 0.24]
        ], dtype=torch.float32)
muE = torch.tensor([0., 0., 0.])  # Drift coefficients for agent endowments
alpha = torch.tensor([0.4, 0.3, 0.3])  # Agent risk preferences/weights
epsilon = 1e-7  # Small epsilon for numerical stability

# Neural network and training configuration

def generate(num_paths, num_steps, dimension, terminal_time, device='cpu'):
    """Generate discrete Brownian motion paths.
    
    Args:
        num_paths: Number of trajectories (M)
        num_steps: Number of time steps (N)
        dimension: Dimension of Brownian motion (D)
        terminal_time: Terminal time T
        device: PyTorch device
        
    Returns:
        t: Time grid of shape (M, N+1, 1)
        W: Brownian paths of shape (M, N+1, D)
    """
    dt = terminal_time / num_steps
    
    # Time increments and Brownian increments
    Dt = np.zeros((num_paths, num_steps + 1, 1))
    DW = np.zeros((num_paths, num_steps + 1, dimension))
    
    Dt[:, 1:, :] = dt
    DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(num_paths, num_steps, dimension))
    
    # Cumulative sums to get paths
    t = np.cumsum(Dt, axis=1)
    W = np.cumsum(DW, axis=1)
    
    return (torch.from_numpy(t).float().to(device), 
            torch.from_numpy(W).float().to(device))

def get_analytical_theta(sigD, sigE, alpha):
    """
    Analytical Radner equilibrium strategies.
    Returns:
        Tensor of shape (I,) : theta^i
    """
    b0 = sigD.squeeze(0)                # (D,)
    b_agents = sigE # (I, D)

    # numerator common term
    sum_b = b0 + (alpha.unsqueeze(1) * b_agents).sum(0)
    common = torch.dot(sum_b, b0)

    denom = torch.dot(b0, b0)

    theta = (common - torch.mv(b_agents, b0)) / denom
    return theta  # (I,)


def S_exact(sigD, sigE, alpha, t,X): #K*1, K*D       
    """
    Compute the exact solution S(t,W) of the Radner equilibrium FBSDE.
    
    Parameters:
    t (float): time
    X (torch.tensor): Brownian motion with shape (M,N,D)
    
    Returns:
    torch.tensor: exact solution S(t,W) with shape (M,N,I)
    """
    I = alpha.shape[0]
    temp = sigD
   
    for i in range(I):
        temp=temp+alpha[i]*sigE[i,:]
    a=torch.sum(temp*sigD,-1,keepdims=True)
    return (t-1)*a+torch.sum(sigD*X, -1, keepdims=True) #K*1


def S_path(sigD, sigE, alpha, t_star, W_star):
    """
    Generate exact stock price paths

    Parameters
    ----------
    t_star : (K, N+1, 1)
    W_star : (K, N+1, D)

    Returns
    -------
    S_path : (K, N+1, 1)
    """

    K, NT, _ = t_star.shape  # NT = N+1

    S_list = []

    for n in range(NT):
        tn = t_star[:, n, :].float()   # (K,1)
        Wn = W_star[:, n, :].float()   # (K,D)

        Sn = S_exact(sigD, sigE, alpha, tn, Wn)
        S_list.append(Sn)

    S_path = torch.stack(S_list, dim=1)  # (K, N+1, 1)

    return S_path



t, W = BrownianMotionGenerator.generate(
            1, N, D, T
        )



def plot_S_theta_rho(sigD, alpha, t_grid, W_path, rho_list=(-0.9, -0.2, 0.0, 0.2, 0.9), save_path=None):
    """
    Plot S_t paths and theta^i vs rho in a 2x2 grid figure.
    """
    I = alpha.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes = axes.flatten()  # flatten for easier indexing

    # --------------------------
    # 1. Plot S(t) paths (top-left)
    # --------------------------
    ax = axes[0]
    for r in rho_list:
        sigE = torch.tensor([
            [0.3 * r, 0.3 * np.sqrt(1 - r**2), 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0]
        ], dtype=torch.float32)
        S_vals = S_path(sigD, sigE, alpha, t_grid, W_path)  # (K, N+1, 1)
        
        # squeeze batch and last dimension
        S_vals = S_vals.squeeze(-1)  # (K, N+1)
        
        # 对 K 条轨迹取均值
        S_mean = S_vals.mean(axis=0)  # (N+1,)
        
        # 取 t_grid 的第 0 条轨迹作为 x
        t = t_grid[0, :, 0].numpy()  # (N+1,)
        
        ax.plot(t, S_mean, lw=2, label=rf"$\rho={r}$")
    
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$S_t$")
    ax.set_title("Stock price paths")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --------------------------
    # 2. Plot theta^i vs rho (top-right, bottom-left, bottom-right)
    # --------------------------
    theta_all = np.zeros((len(rho_list), I))
    for j, r in enumerate(rho_list):
        sigE = torch.tensor([
            [0.3 * r, 0.3 * np.sqrt(1 - r**2), 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0]
        ], dtype=torch.float32)
        theta = get_analytical_theta(sigD, sigE, alpha)
        theta_all[j, :] = theta.detach().numpy()

    for i in range(I):
        ax = axes[i+1]
        ax.plot(rho_list, theta_all[:, i], marker='o', lw=2)
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(rf"$\theta^{i+1}$")
        ax.set_title(rf"Agent {i+1} $\theta^{{{i+1}}}$ vs $\rho$")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
            plt.savefig(save_path, bbox_inches="tight")


    plt.show()



# def plot_S_theta_alpha(sigD, sigE, t_grid, W_path, alpha_list=(0.2, 0.4, 0.8), save_path=None):
#     """
#     Plot S_t paths and theta^i vs rho in a 2x2 grid figure.
#     """
    
#     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
#     axes = axes.flatten()  # flatten for easier indexing

#     # --------------------------
#     # 1. Plot S(t) paths (top-left)
#     # --------------------------
#     ax = axes[0]
#     for a in alpha_list:
#         alpha = torch.tensor([a, (1-a)/2, (1-a)/2], dtype=torch.float32)
#         I = alpha.shape[0]
#         S_vals = S_path(sigD, sigE, alpha, t_grid, W_path)  # (K, N+1, 1)
        
#         # squeeze batch and last dimension
#         S_vals = S_vals.squeeze(-1)  # (K, N+1)
        
#         # 对 K 条轨迹取均值
#         S_mean = S_vals.mean(axis=0)  # (N+1,)
        
#         # 取 t_grid 的第 0 条轨迹作为 x
#         t = t_grid[0, :, 0].numpy()  # (N+1,)
        
#         ax.plot(t, S_mean, lw=2, label=rf"$\alpha={a}$")
    
#     ax.set_xlabel(r"$t$")
#     ax.set_ylabel(r"$S_t$")
#     ax.set_title("Stock price paths")
#     ax.grid(True, alpha=0.3)
#     ax.legend()

#     # --------------------------
#     # 2. Plot theta^i vs rho (top-right, bottom-left, bottom-right)
#     # --------------------------
#     theta_all = np.zeros((len(alpha_list), I))
#     for j, r in enumerate(alpha_list):
#         alpha = torch.tensor([r, (1-r)/2, (1-r)/2], dtype=torch.float32)
#         theta = get_analytical_theta(sigD, sigE, alpha)
#         theta_all[j, :] = theta.detach().numpy()

#     for i in range(I):
#         ax = axes[i+1]
#         ax.plot(alpha_list, theta_all[:, i], marker='o', lw=2)
#         ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)
#         ax.set_xlabel(r"$\rho$")
#         ax.set_ylabel(rf"$\theta^{i+1}$")
#         ax.set_title(rf"Agent {i+1} $\theta^{{{i+1}}}$ vs $\alpha$")
#         ax.grid(True, alpha=0.3)

#     plt.tight_layout()

#     if save_path:
#             plt.savefig(save_path, bbox_inches="tight")


#     plt.show()

    

plot_S_theta_rho(
    sigD,
    alpha,
    t,
    W,
    rho_list=(-0.9, -0.2, 0.0, 0.5, 1.0), 
    save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/path_theta1.pdf",
)


# plot_S_theta_alpha(
#     sigD,
#     sigE,
#     t,
#     W,
#     alpha_list=(0.2, 0.4, 0.8), 
#     save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/path_theta2.pdf",
# )