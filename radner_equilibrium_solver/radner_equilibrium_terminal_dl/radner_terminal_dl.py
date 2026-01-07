"""
radner_terminal_dl.py

Module for learning Radner equilibria with neural networks using the defination of the Radner equilibrium directly.

Contains:
- SimpleNet: small fully-connected Tanh network with orthogonal initialization.
- RadnerEquilibriumSolver: end-to-end learning setup that concurrently learns the
  price process S(t,W) and agent strategies θ^i(t,W,S) by directly optimizing the
  Radner equilibrium conditions (agent utility maximization, market clearing,
  terminal/anchor conditions and a supermartingale constraint).

This file is intended to run as a script (see the __main__ block) and exposes a
trainer-style class with .train(), .evaluate(), and .plot_results() helpers.

Notes:
- Uses PyTorch; device selected automatically (CUDA if available).
- Key hyperparameters are set in RadnerEquilibriumSolver.__init__.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =====================================
# Network definitions
# =====================================
class SimpleNet(nn.Module):
    """A small fully-connected neural network with Tanh activations.

    Architecture:
      Linear(in_dim -> hidden) -> Tanh -> (depth-1) * [Linear(hidden->hidden) -> Tanh] -> Linear(hidden -> out_dim)

    Initialization:
      Linear weights are initialized orthogonally with gain=0.5; biases set to zero.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        hidden (int): hidden layer width (default: 128)
        depth (int): number of linear blocks including input and output layers (default: 3)
    """
    def __init__(self, in_dim, out_dim, hidden=128, depth=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.Tanh())
        
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        
        # small initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): shape (..., in_dim)

        Returns:
            Tensor: shape (..., out_dim)
        """
        return self.net(x)


# =====================================
# Radner equilibrium solver
# =====================================
class RadnerEquilibriumSolver:
    """
    Solver that learns Radner equilibrium quantities from simulated Brownian paths.

    High-level algorithm:
      1. Simulate M Brownian paths (D-dimensional) over N time steps.
      2. Parameterize price S(t,W) with S_net and each agent's trading policy θ^i(t,W,S)
         with per-agent theta_nets.
      3. Compute utilities, Radner supermartingale constraint (enforced as penalty),
         market clearing penalty, terminal price penalty and S0 anchor penalty.
      4. Optimize neural nets jointly by minimizing the composite loss.

    Important attributes (examples):
      - T, N, M, dt: time horizon and discretization
      - D, I: Brownian dimension and number of agents
      - alpha: agent weights used in market clearing
      - B: vectors used to define terminal payoffs Xi and E^i
      - S_net: SimpleNet mapping (t, W_t) -> S_t
      - theta_nets: ModuleList of SimpleNet mapping (t, W_t, S_t) -> θ^i_t
      - opt, scheduler: optimizer and LR scheduler

    Typical usage:
      solver = RadnerEquilibriumSolver(...)
      solver.train(steps=3000)
      results = solver.evaluate(M=1000)
      solver.plot_results(results)

    Args:
        T (float): time horizon
        N (int): number of intervals (discrete steps = N, grid size N+1)
        M (int): batch size / number of simulated paths
        D (int): Brownian motion dimension
        I (int): number of agents
        device (torch.device | None): torch device; auto-detects CUDA if None
    """
    def __init__(
        self,
        T=1.0,
        N=100,
        M=512,
        D=4,
        I=3,
        device=None
    ):
        self.T, self.N, self.M = T, N, M
        self.dt = T / N
        self.D, self.I = D, I
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # market parameters
        self.alpha = torch.tensor([0.4, 0.3, 0.3])
        self.B = torch.stack([
            torch.tensor([0.3, 0.0, 0.1, 0.0]),  # b0: Xi = b0 · W_T
            torch.tensor([0.3, 0.3, 0.0, 0.0]),  # b1: E^1
            torch.tensor([0.2, 0.0, 0.3, 0.0]),  # b2: E^2
            torch.tensor([0.1, 0.0, 0.0, 0.3]),  # b3: E^3
        ])
        
       
        # Neural networks
        # S_net: (t, W) -> S_t

        # --- networks ---
    


        self.S_net = SimpleNet(1 + D, 1, hidden=128, depth=3).to(self.device)
        
        # θ_nets: (t, W, S) -> θ^i_t  (strategies depend on current price)
        self.theta_nets = nn.ModuleList([
            SimpleNet(1 + D + 1, 1, hidden=64, depth=2).to(self.device)
            for _ in range(I)
        ])
    
        # optimizer
        params = list(self.S_net.parameters())
        for net in self.theta_nets:
            params += list(net.parameters())
        
        self.opt = optim.Adam(params, lr=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=3000)
        
        # training history
        self.history = {
            'loss': [], 'util': [], 'clear': [], 
            'mse_vs_analytical': []
        }
    
    def generate_paths(self, M):
        """Simulate M Brownian motion paths.

        Returns:
            t (Tensor): shape (M, N+1, 1) time grid
            W (Tensor): shape (M, N+1, D) Brownian paths
            dW (Tensor): shape (M, N, D) Brownian increments
        """
        dW = torch.randn(M, self.N, self.D, device=self.device) * np.sqrt(self.dt)
        W = torch.zeros(M, self.N + 1, self.D, device=self.device)
        W[:, 1:] = torch.cumsum(dW, dim=1)
        
        t = torch.linspace(0, self.T, self.N + 1, device=self.device)
        t = t.view(1, -1, 1).expand(M, -1, -1)
        
        return t, W, dW
    
    def get_analytical_price(self, t, W):
        """
        Compute the analytical price used for monitoring.

        Analytical form used in this project:
          S(t,W) = (t - T) * coeff + W · b0
        where coeff = (b0 + Σ α_i b_{i+1}) · b0

        Args:
            t (Tensor): (M, N+1, 1)
            W (Tensor): (M, N+1, D)

        Returns:
            Tensor: S_analytical shape (M, N+1)
        """
        sum_b = self.B[0] + (self.alpha.unsqueeze(1) * self.B[1:]).sum(0)
        coeff = (sum_b @ self.B[0]).item()
        
        # S(t,W) = (t - 1) * coeff + W @ b0
        S_analytical = (t.squeeze(-1) - 1.0) * coeff + (W @ self.B[0])
        return S_analytical
    
    def forward(self, t, W):
        """
        Compute learned price paths and agent strategies for given states.

        Args:
            t (Tensor): (M, N+1, 1) time grid
            W (Tensor): (M, N+1, D) Brownian paths

        Returns:
            S_learned (Tensor): (M, N+1) learned prices
            thetas (Tensor): (I, M, N) learned strategies per agent per time step (excluding terminal)
        """
        M, N1, _ = t.shape
        
        # 1. Learn price S(t, W)
        S_learned = []
        for n in range(N1):
            t_n = t[:, n, :]  # (M, 1)
            W_n = W[:, n, :]  # (M, D)
            x = torch.cat([t_n, W_n], dim=1)  # (M, 1+D)
            S_n = self.S_net(x).squeeze(-1)  # (M,)
            S_learned.append(S_n)
        
        S_learned = torch.stack(S_learned, dim=1)  # (M, N+1)
        
        # 2. Learn strategies θ^i(t, W, S)
        thetas = []
        for n in range(N1 - 1):  # only need strategies for N time steps
            t_n = t[:, n, :]
            W_n = W[:, n, :]
            S_n = S_learned[:, n].unsqueeze(-1)  # (M, 1)
            
            x = torch.cat([t_n, W_n, S_n], dim=1)  # (M, 1+D+1)
            
            theta_n = []
            for i, net in enumerate(self.theta_nets):
                theta_i = net(x).squeeze(-1)  # (M,)
                # clamp range
                theta_i = torch.tanh(theta_i) * 5.0
                theta_n.append(theta_i)
            
            thetas.append(torch.stack(theta_n, dim=0))  # (I, M)
        
        thetas = torch.stack(thetas, dim=-1)  # (I, M, N)
        
        return S_learned, thetas
    
    def compute_loss(self, t, W, S_learned, thetas):
        """
        Compute the composite training loss used to enforce equilibrium.

        Components (and where to find their coefficients in code):
          - util_loss: negative expected exponential utility (summed across agents)
          - sm_constraint_loss: Radner supermartingale violation (penalized heavily)
          - clear_loss: market clearing MSE (weighted by 500.0)
          - terminal_loss: penalty for terminal price mismatch with Xi
          - S0_loss: anchor for initial price using state-price density
          - price_smooth, theta_reg: regularization terms

        Returns:
            dict: breakdown {'total', 'util', 'util_sum', 'sm', 'clear', 'terminal', 'smooth'}
        """
        M = S_learned.shape[0]

        # price increments
        dS_learned = S_learned[:, 1:] - S_learned[:, :-1]  # (M, N)

        # terminal payoff
        Xi = (W[:, -1] * self.B[0]).sum(-1)
        E = [(W[:, -1] * self.B[1 + i]).sum(-1) for i in range(self.I)]

        util_loss = 0.0
        sm_constraint_loss = 0.0   # ⭐ Radner constraint

        utilities = []

        for i in range(self.I):
            # ∫ θ dS
            pnl = (thetas[i] * dS_learned).sum(1)  # (M,)
            terminal_wealth = pnl + E[i]

            # ===== utility =====
            utility = -torch.exp(-terminal_wealth)
            utilities.append(utility.mean())
            util_loss -= utility.mean()

            # ===== Radner supermartingale constraint =====
            # density dQ^i/dP ∝ exp(-∫θdS - E)
            weight = torch.exp(-terminal_wealth).detach()

            # E_P[(∫θ dS) * density] ≤ 0
            violation = (pnl * weight).mean()

            sm_constraint_loss += torch.relu(violation)

        # ===== Market clearing =====
        weighted = (self.alpha.view(self.I, 1, 1) * thetas).sum(0)
        clear_loss = ((weighted - 1.0) ** 2).mean()

        # ===== Terminal condition =====
        terminal_loss = ((S_learned[:, -1] - Xi) ** 2).mean()

        # ===== S0 pricing anchor =====
        with torch.no_grad():
            # state price density (up to normalization)
            spd = torch.exp(
                -Xi - sum(self.alpha[i] * E[i] for i in range(self.I))
            )
            spd = spd / spd.mean()

        S0_target = (spd * Xi).mean()
        S0_learned = S_learned[:, 0].mean()

        S0_loss = (S0_learned - S0_target) ** 2

        # ===== Regularization terms =====
        price_smooth = ((S_learned[:, 2:] - 2*S_learned[:, 1:-1] + S_learned[:, :-2]) ** 2).mean()
        theta_reg = sum((thetas[i] ** 2).mean() for i in range(self.I)) * 0.01

        # ===== Total loss =====
        loss = (
            util_loss
            + 200.0 * sm_constraint_loss     # ⭐ core term
            + 500.0 * clear_loss
            + 200.0 * terminal_loss
            + 200.0  * S0_loss      # ⭐ added
            + 0.5 * price_smooth
            + theta_reg
        )

        return {
            'total': loss,
            'util': util_loss,
            'util_sum': sum(utilities),
            'sm': sm_constraint_loss,
            'clear': clear_loss,
            'terminal': terminal_loss,
            'smooth': price_smooth
        }
    
    def train(self, steps=3000, print_every=200):
        """Training loop.

        Args:
            steps (int): number of optimization steps
            print_every (int): logging frequency
        """
        print("="*80)
        print("Radner equilibrium deep learning solver (learning directly from equilibrium definitions)")
        print("="*80)
        print(f"{'Iter':<8} {'Loss':<10} {'Utility':<10} {'Clearing':<12} {'Terminal':<12}")
        print("-"*80)
        
        for it in range(1, steps + 1):
            # generate data
            t, W, dW = self.generate_paths(self.M)
            
            # learning (do not use analytical solution for training)
            S_learned, thetas = self.forward(t, W)
            
            # compute loss (excluding analytical monitoring terms)
            losses = self.compute_loss(t, W, S_learned, thetas)
            
            # backward
            self.opt.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.opt.param_groups[0]['params'], 1.0)
            self.opt.step()
            self.scheduler.step()
            
            # logging
            self.history['loss'].append(losses['total'].item())
            self.history['util'].append(losses['util_sum'].item())
            self.history['clear'].append(losses['clear'].item())
            
            # compute comparison with analytical solution for monitoring (not used for training)
            if it % print_every == 0 or it == 1:
                with torch.no_grad():
                    S_analytical = self.get_analytical_price(t, W)
                    mse_vs_analytical = ((S_learned - S_analytical) ** 2).mean().item()
                    self.history['mse_vs_analytical'].append(mse_vs_analytical)
                
                print(f"{it:<8} {losses['total'].item():<10.4f} "
                      f"{losses['util_sum'].item():<10.6f} "
                      f"{losses['clear'].item():<12.2e} "
                      f"{losses['terminal'].item():<12.2e}")
                print(f"         [Monitor] MSE vs Analytical = {mse_vs_analytical:.6f}")
        
        print("="*80)
        print("Training complete!")
        print("="*80)
    
    @torch.no_grad()
    def evaluate(self, M=1000):
        """Evaluate learned models on M simulated paths and print diagnostics.

        Returns:
            dict: keys include 'S_learned', 'S_analytical', 'thetas', 't', 'W', 'losses', 'mse_price'
        """
        t, W, dW = self.generate_paths(M)
        
        # learned outputs
        S_learned, thetas = self.forward(t, W)
        
        # analytical solution
        S_analytical = self.get_analytical_price(t, W)
        
        # compute loss
        losses = self.compute_loss(t, W, S_learned, thetas)
        
        # price comparison stats
        price_diff = S_learned - S_analytical
        mse_price = (price_diff ** 2).mean().item()
        mae_price = price_diff.abs().mean().item()
        max_price_error = price_diff.abs().max().item()
        correlation = torch.corrcoef(torch.stack([
            S_learned.flatten(), 
            S_analytical.flatten()
        ]))[0, 1].item()
        
        # market clearing stats
        weighted = (self.alpha.view(self.I, 1, 1) * thetas).sum(0)
        clearing_error = (weighted - 1.0).abs()
        
        print("\n" + "="*70)
        print("Evaluation results")
        print("="*70)
        print("\n[Equilibrium conditions]")
        print(f"  Mean utility:           {losses['util_sum'].item():.6f}")
        print(f"  Market clearing MSE:    {losses['clear'].item():.2e}")
        print(f"  Max clearing error:     {clearing_error.max().item():.6f}")
        print(f"  Terminal condition MSE: {losses['terminal'].item():.2e}")
        
        print("\n[Learned price vs Analytical]")
        print(f"  MSE:                    {mse_price:.6f}")
        print(f"  MAE:                    {mae_price:.6f}")
        print(f"  Max error:              {max_price_error:.6f}")
        print(f"  Correlation:            {correlation:.6f}")
        
        # error by time
        mse_by_time = (price_diff ** 2).mean(0)
        print(f"\n  Initial time error:     {mse_by_time[0].item():.6f}")
        print(f"  Middle time avg error:  {mse_by_time[1:-1].mean().item():.6f}")
        print(f"  Terminal time error:    {mse_by_time[-1].item():.6f}")
        
        # strategy statistics
        print("\n[Strategy statistics]")
        theoretical = 1.0 / self.alpha.cpu()
        for i in range(self.I):
            mean_theta = thetas[i].mean().item()
            std_theta = thetas[i].std().item()
            print(f"  Agent {i+1}: mean={mean_theta:.4f} (theoretical={theoretical[i].item():.4f}), "
                  f"std={std_theta:.4f}")
        
        print("="*70)
        
        return {
            'S_learned': S_learned.cpu(),
            'S_analytical': S_analytical.cpu(),
            'thetas': thetas.cpu(),
            't': t.cpu(),
            'W': W.cpu(),
            'losses': losses,
            'mse_price': mse_price
        }
    
    def plot_results(self, results=None):
        """Plot training history and price/strategy comparisons.

        If results is None, evaluate(M=1000) is called to produce a results dict.
        Returns:
            matplotlib.Figure
        """
        if results is None:
            results = self.evaluate(M=1000)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 1, hspace=0.3, wspace=0.3)
        
        # 1. Training loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.history['loss'])
        ax1.set_title('Total Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_yscale('log')
        ax1.grid(True)
        
        # 4. Price paths comparison - multiple paths
        ax4 = fig.add_subplot(gs[1, 0])
        S_learned = results['S_learned']
        S_analytical = results['S_analytical']
        t = results['t'][0, :, 0].numpy()
        
        num_paths = min(5, S_learned.shape[0])
        for i in range(num_paths):
            ax4.plot(t, S_learned[i], '-', alpha=0.7, linewidth=2, label=f'Learned {i+1}')
            ax4.plot(t, S_analytical[i], '--', alpha=0.7, linewidth=2, label=f'Analytical {i+1}')
        ax4.set_title('Price Paths: Learned vs Analytical')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Price')
        ax4.legend(fontsize=7, ncol=2)
        ax4.grid(True)
        
        plt.suptitle('Radner Equilibrium: Deep Learning vs Analytical Solution', 
                    fontsize=14, y=0.995)
        
        return fig


# =====================================
# Run as script
# =====================================
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    # create solver
    solver = RadnerEquilibriumSolver(
        T=1.0,
        N=100,
        M=512,
        D=4,
        I=3
    )
    
    # train
    solver.train(steps=3000, print_every=200)
    
    # evaluate
    results = solver.evaluate(M=1000)
    
    # visualize
    fig = solver.plot_results(results)
    plt.show()