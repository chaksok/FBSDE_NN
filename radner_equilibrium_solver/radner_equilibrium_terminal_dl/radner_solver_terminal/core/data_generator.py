import numpy as np
import torch


class BrownianMotionGenerator:
    """Generate Brownian motion paths for FBSDE training."""
    
    @staticmethod
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
