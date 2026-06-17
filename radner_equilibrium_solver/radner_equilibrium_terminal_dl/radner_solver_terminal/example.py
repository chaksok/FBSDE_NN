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
# import plotly.express as px
sns.set()
# plt.style.use('ggplot')

# Import custom modules
from models.radner_terminal_BSDE_solver import RadnerNeuralNetwork2
from utils.visualization import RadnerVisualizer
from core.data_generator import BrownianMotionGenerator

# Set random seeds for reproducibility
torch.manual_seed(100)
np.random.seed(100)

    
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
drift_D = 0.2  # Drift coefficient for dividend process
sigD = torch.tensor([[0.3, 0.0, 0.1, 0.0]])  # Volatility matrix for dividend
muE = torch.tensor([0.1, 0.1, 0.1])  # Drift coefficients for agent endowments
sigE = torch.tensor([
    [0.3, 0.3, 0.0, 0.0],  # Volatility for agent 1 endowment
    [0.2, 0.0, 0.3, 0.0],  # Volatility for agent 2 endowment
    [0.1, 0.0, 0.0, 0.3]   # Volatility for agent 3 endowment
])
alpha = torch.tensor([0.4, 0.3, 0.3])  # Agent risk preferences/weights
epsilon = 1e-7  # Small epsilon for numerical stability

# Training hyperparameters
epoch = 10  # Number of training epochs
NIter = 1000  # Number of iterations per epoch
patience = 2  # Patience for early stopping


t_train, W_train = BrownianMotionGenerator.generate(M * NIter, N, D, T)
t_valid, W_valid = BrownianMotionGenerator.generate(1000, 200, D, T, device='cpu')

# Neural network and training configuration
config = {
    'T': T,
    'M': M,
    'N': N,
    'D': D,
    'I': I,
    'K': K,
    'layers': [D + 1] + 4 * [256] + [(I + 1) * (1)],  # Input: time + state, Output: Y and Z for all processes
    # 'layers_z': [D + 1] + 4 * [256] + [(I + 1) * (D)],  #NN layers
    # 'layers_y': [D + 1] + 4 * [256] + [(I + 1) * 1],  #NN layers
    'learning_rate': 1e-4,
    'device': 'cpu',
    't_valid': t_valid,
    'W_valid': W_valid,
    'checkpoint_path': '/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/best_model.pt',
    'drift_D': drift_D,
    'sigD': sigD,
    'muE': muE,
    'sigE': sigE,
    'alpha': alpha,
    'epsilon': epsilon,
    't_train': t_train,  # To be set after generating training data
    'W_train': W_train  # To be set after generating training data
}


print("\nProblem Configuration:")
print(f"  Terminal time: {config['T']}")
print(f"  State dimension: {config['D']}")
print(f"  Number of agents: {config['I']}")
print(f"  Time steps: {config['N']}")
print(f"  Batch size: {config['M']}")
# print(f"  Network architecture: {config['layers']}")
# print(f"  Network architecture: {config['layers_y']} and {config['layers_z']}")

# # Initialize solver
# print("\nInitializing solver...")
# solver1 = RadnerNeuralNetwork4(config)

# print("\nStarting training...")
# solver1.train(NIter, epoch, patience)     


# # =============================================================================
# # Training Summary
# # =============================================================================

# # Print training summary
# print("\n" + "="*60)
# print("Training Summary:")
# print("="*60)
# print(f"Best epoch: {solver1.history['best_epoch'] + 1}")
# print(f"Best validation loss: {solver1.history['best_val_loss']:.6f}")
# print(f"Final training loss: {solver1.history['train_loss'][-1]:.6f}")
# print(f"Total training time: {solver1.history['train_time']:.2f} seconds")
# print(f"Average time per epoch: {solver1.history['time_per_epoch']:.2f} seconds")

# # =============================================================================
# # Prediction and Visualization
# # =============================================================================

# Generate predictions and visualize
# print("\n" + "="*60)
# print("Generating Predictions and Visualizations...")
# print("="*60)

# # Predict using the trained model
# Y_pred, Y_exact, Theta_pred, Theta_real = solver1.predict()

# # Initialize visualizer
# viz = RadnerVisualizer()

# # Plot predicted vs exact paths
# viz.plot_paths(
#     Y_pred,
#     Y_exact,
#     save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/model4/path.pdf",
# )

# viz.plot_price_and_theta(
#     Y_pred,
#     Y_exact,
#     Theta_pred,
#     Theta_real,
#     save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/model4/theta_path.pdf",
# )


# # Plot relative errors
# viz.plot_rmse_errors(
#     Y_pred,
#     Y_exact,
#     save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/model4/rmse_error.pdf",
# )

# viz.plot_relative_errors_theta(
#         Theta_pred,
#         Theta_real,
#         save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/model4/relative_error.pdf",
#         max_cols=3
#     )

from utils.visualization import plot_loss_decay_MN

# fix the path numbers M
M_list = [20, 100]
N_list = [20, 100, 200]

loss_records = {}

for N_ in N_list:
    M_ = 100
    print("=" * 60)
    print(f"Training with M={M_}, N={N_}")
    print("=" * 60)

    config_tmp = config.copy()
    config_tmp["M"] = M_
    config_tmp["N"] = N_

    solver = RadnerNeuralNetwork2(config_tmp)
    solver.train(NIter, epoch, patience)

    loss_records[(M_, N_)] = solver.history.copy()

import pickle

with open('/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/data/model2/loss_records.pkl', 'wb') as f:
    pickle.dump(loss_records, f)


# with open('/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/data/model1/loss_records.pkl', 'rb') as f:
#     data = pickle.load(f)

# print(data.keys())

# plot_loss_decay_MN(
#     loss_records,
#     save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/model1/train_error_iteration1.pdf",
# )


# # # fix the time steps N
# # loss_records2 = {}

# # for M_ in M_list:
# #     N_ = 100
# #     print("=" * 60)
# #     print(f"Training with M={M_}, N={N_}")
# #     print("=" * 60)

# #     config_tmp = config.copy()
# #     config_tmp["M"] = M_
# #     config_tmp["N"] = N_

# #     solver = RadnerNeuralNetwork1(config_tmp)
# #     solver.train(NIter, epoch, patience)

# #     loss_records2[(M_, N_)] = solver.history.copy()


# # plot_loss_decay_MN(
# #     loss_records2,
# #     save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/model1/train_error_iteration2.pdf",
# # )



print("Done!")

# =============================================================================
# Notes:
# - The script saves plots to the specified paths.
# - Adjust configuration parameters as needed for different problems.
# - Ensure the custom modules (models and utils) are properly installed or in the path.
# =============================================================================