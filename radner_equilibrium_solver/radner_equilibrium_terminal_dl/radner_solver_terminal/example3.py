# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
from core.data_generator import BrownianMotionGenerator
from core.network import FeedForwardNN as myNN
from models.radner_terminal_general_solver import RadnerEquilibriumSolver
from utils.visualization import RadnerVisualizer

torch.manual_seed(1)
np.random.seed(1)

# -------------------------
# 1. Benchmark Nonlinear Economy: Example 4.4
# -------------------------

# Time horizon and discretization
T = 1  # Terminal time
M = 100  # Number of trajectories per batch
N = 100  # Number of time steps
D = 2  # Dimension of Brownian motion (state space)
I = 2  # Number of agents
K = 1000  # Number of validation trajectories

config = {
    'T': T,
    'M': M,
    'N': N,
    'D': D,
    'I': I,
    'K': K,
    'learning_rate': 1e-4,
    'device': 'cpu',
    'layers': [D + 1] + 4 * [256] + [(I + 1) * (1 + D)],  # Input: time + state, Output: Y and Z for all processes
    'drift_D': 0.0,
    'sigD': torch.tensor([[0.2/np.sqrt(2), 0.2/np.sqrt(2)]], dtype=torch.float32),  # dividend depends on W1+W2
    'muE': torch.zeros(2),
    'sigE': torch.tensor([
        [0.0, 0.2/np.sqrt(2)],  # Agent1 exposure to W2 (put)
        [0.0, 0.2/np.sqrt(2)]   # Agent2 opposite
    ], dtype=torch.float32),
    'alpha': torch.tensor([0.7, 0.3]),  # CARA risk aversion
    'epsilon': 1e-6,
    'checkpoint_path': 'radner_example44.pth'
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
# print(f"  Network architecture: {config['layers_y']} and {config['layers_z']}")

# Initialize solver
print("\nInitializing solver...")
solver =RadnerEquilibriumSolver(config)

print("\nStarting training...")
solver.train(NIter, epoch, patience)     


# Print training summary
print("\n" + "="*60)
print("Training Summary:")
print("="*60)
print(f"Best epoch: {solver.history['best_epoch'] + 1}")
print(f"Best validation loss: {solver.history['best_val_loss']:.6f}")
print(f"Final training loss: {solver.history['train_loss'][-1]:.6f}")
print(f"Total training time: {solver.history['train_time']:.2f} seconds")
print(f"Average time per epoch: {solver.history['time_per_epoch']:.2f} seconds")

# Generate predictions and visualize
print("\n" + "="*60)
print("Generating Predictions and Visualizations...")
print("="*60)

t, W = solver.t_star, solver.W_star

Y_path, Z_path, theta_path = solver.predict(t, W)

# -------------------------
# 5. Visualization
# -------------------------


viz = RadnerVisualizer()

# Plot predicted 

viz.plot_price_and_theta_pred(
    Y_path,
    theta_path,
    save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/theta_path.pdf",
)

print("Done!")
# # 取固定 W1 = 0，查看 S vs W2
# W2_vals = torch.linspace(-3, 3, 50)
# S_vals = []
# theta1_vals = []
# theta2_vals = []

# for w2 in W2_vals:
#     W_temp = torch.zeros((1, 1, 2))
#     W_temp[0, 0, 0] = 0.0       # W1 = 0
#     W_temp[0, 0, 1] = w2        # W2 = w2

#     S_pred, _, theta_pred, _ = solver.net_u(W_temp[:,0,:], W_temp[:,0,:])
#     S_vals.append(S_pred[0,0].item())      # asset price
#     theta1_vals.append(theta_pred[0][0,0].item())
#     theta2_vals.append(theta_pred[1][0,0].item())

# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(W2_vals.numpy(), S_vals, lw=2)
# plt.xlabel("W2")
# plt.ylabel("S(t,W2)")
# plt.title("Asset Price vs Weather Risk W2")
# plt.grid(True)

# plt.subplot(1,2,2)
# plt.plot(W2_vals.numpy(), theta1_vals, label="Agent 1")
# plt.plot(W2_vals.numpy(), theta2_vals, label="Agent 2")
# plt.xlabel("W2")
# plt.ylabel("Portfolio θ")
# plt.title("Agent Portfolios vs Weather Risk W2")
# plt.legend()
# plt.grid(True)
# plt.show()