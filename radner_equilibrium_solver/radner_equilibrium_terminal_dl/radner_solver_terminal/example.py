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
drift_D = 0.2  # Drift coefficient for dividend process
sigD = torch.tensor([[0.3, 0.0, 0.1, 0.0]])  # Volatility matrix for dividend
muE = torch.tensor([0.1, 0.1, 0.1])  # Drift coefficients for agent endowments
sigE = torch.tensor([
    [0.3, 0.3, 0.0, 0.0],  # Volatility for agent 1 endowment
    [0.2, 0.0, 0.3, 0.0],  # Volatility for agent 2 endowment
    [0.1, 0.0, 0.0, 0.3]   # Volatility for agent 3 endowment
])
alpha = [0.4, 0.3, 0.3]  # Agent risk preferences/weights
epsilon = 1e-7  # Small epsilon for numerical stability

# Neural network and training configuration
config = {
    'T': T,
    'M': M,
    'N': N,
    'D': D,
    'I': I,
    'K': K,
    'layers': [D + 1] + 4 * [256] + [(I + 1) * (1 + D)],  # Input: time + state, Output: Y and Z for all processes
    'learning_rate': 1e-4,
    'device': 'cpu',
    'checkpoint_path': '/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/best_model.pt',
    'drift_D': drift_D,
    'sigD': sigD,
    'muE': muE,
    'sigE': sigE,
    'alpha': alpha,
    'epsilon': epsilon
}

# Training hyperparameters
epoch = 50  # Number of training epochs
NIter = 1000  # Number of iterations per epoch
patience = 2  # Patience for early stopping

print("\nProblem Configuration:")
print(f"  Terminal time: {config['T']}")
print(f"  State dimension: {config['D']}")
print(f"  Number of agents: {config['I']}")
print(f"  Time steps: {config['N']}")
print(f"  Batch size: {config['M']}")
print(f"  Network architecture: {config['layers']}")

# Initialize solver
print("\nInitializing solver...")
solver1 = RadnerNeuralNetwork1(config)

print("\nStarting training...")
solver1.train(NIter, epoch, patience)     


# =============================================================================
# Training Summary
# =============================================================================

# Print training summary
print("\n" + "="*60)
print("Training Summary:")
print("="*60)
print(f"Best epoch: {solver1.history['best_epoch'] + 1}")
print(f"Best validation loss: {solver1.history['best_val_loss']:.6f}")
print(f"Final training loss: {solver1.history['train_loss'][-1]:.6f}")

# =============================================================================
# Prediction and Visualization
# =============================================================================

# Generate predictions and visualize
print("\n" + "="*60)
print("Generating Predictions and Visualizations...")
print("="*60)

# Predict using the trained model
Y_pred, Y_exact = solver1.predict()

# Initialize visualizer
viz = RadnerVisualizer()

# Plot predicted vs exact paths
viz.plot_paths(
    Y_pred,
    Y_exact,
    save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/path.pdf",
)

# Plot relative errors
viz.plot_relative_errors(
    Y_pred,
    Y_exact,
    zeta=0.1,
    save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/error.pdf",
)

print("Done!")

# =============================================================================
# Notes:
# - The script saves plots to the specified paths.
# - Adjust configuration parameters as needed for different problems.
# - Ensure the custom modules (models and utils) are properly installed or in the path.
# =============================================================================