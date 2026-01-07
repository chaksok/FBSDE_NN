import torch
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
from models.radner_terminal_BSDE_solver import RadnerNeuralNetwork1
from utils.visualization import RadnerVisualizer
torch.manual_seed(1)
np.random.seed(1)

    
print("="*60)
print("Radner Equilibrium FBSDE Solver")
print("="*60)

# Create configuration
T=1 #terminal time
M=100  #number of trajectoris
N=100  #number of time step
D=4
I=3
K=1000

# Define problem parameters
drift_D = 0.2
sigD = torch.tensor([[0.3, 0.0, 0.1, 0.0]])
muE = torch.tensor([0.1, 0.1, 0.1])
sigE = torch.tensor([
    [0.3, 0.3, 0.0, 0.0],
    [0.2, 0.0, 0.3, 0.0],
    [0.1, 0.0, 0.0, 0.3]
])
alpha = [0.4, 0.3, 0.3]
epsilon = 1e-7


config = {
    'T': 1.0,
    'M': 100,
    'N': 100,
    'D': 4,
    'I': 3,
    'K': 1000,
    'layers': [D+1]+4*[256]+[(I+1)*(1+D)],  
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


epoch=50
NIter=1000 #number of iteration
patience=2

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


# Print training summary
print("\n" + "="*60)
print("Training Summary:")
print("="*60)
print(f"Best epoch: {solver1.history['best_epoch'] + 1}")
print(f"Best validation loss: {solver1.history['best_val_loss']:.6f}")
print(f"Final training loss: {solver1.history['train_loss'][-1]:.6f}")

# Generate predictions and visualize
print("\n" + "="*60)
print("Generating Predictions and Visualizations...")
print("="*60)

Y_pred, Y_exact = solver1.predict()

# Initialize visualizer
viz = RadnerVisualizer()

viz.plot_paths(
    Y_pred,
    Y_exact,
    save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/path.pdf",
)

viz.plot_relative_errors(
    Y_pred,
    Y_exact,
    zeta = 0.1,
    save_path="/Users/sokchak/Desktop/FBSDE_NN/radner_equilibrium_solver/radner_equilibrium_terminal_dl/radner_solver_terminal/pic/error.pdf",
)

print("Done!")