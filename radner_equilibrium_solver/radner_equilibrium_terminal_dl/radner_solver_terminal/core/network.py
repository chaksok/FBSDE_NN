import torch
from torch import nn


class FeedForwardNN(nn.Module):
    """Feedforward neural network with GELU activations.
    
    Args:
        layers: List of layer sizes, e.g., [5, 256, 256, 20]
    """
    def __init__(self, layers):
        super().__init__()
        self.linear_gelu_stack = nn.Sequential()
        for i in range(len(layers) - 2):
            self.linear_gelu_stack.add_module(f"linear{i}", nn.Linear(layers[i], layers[i + 1]))
            self.linear_gelu_stack.add_module(f'activation{i}', nn.GELU())
        self.linear_gelu_stack.add_module("flinear", nn.Linear(layers[len(layers) - 2], layers[len(layers) - 1]))

    
    def forward(self, t, x):
        """Forward pass: concatenate time and state, then pass through network.
        
        Args:
            t: Time tensor of shape (batch_size, 1)
            x: State tensor of shape (batch_size, D)
            
        Returns:
            Network output of shape (batch_size, output_dim)
        """
        return self.linear_gelu_stack(torch.cat((t, x), 1))
    
    