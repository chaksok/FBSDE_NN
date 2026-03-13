"""
Multi-stock Radner Equilibrium Solver
Based on Theorem 3.3.1: M risky assets, I agents, d Brownian motions (S <= d)

Key structural changes vs 1-stock code
---------------------------------------
Dimension layout of Z (the diffusion block):

    1-stock:  Z  has shape (batch, (1+I)*D)
              Z[:,  0 :  D]  = zeta = sigma  (1-stock vol, shape D)
              Z[:,  D : 2D]  = Z^1           (agent 1)
              ...

    S-stock:  Z  has shape (batch, (S+I)*D)
              Z[:,    0 : S*D]       = sigma  (S-stock vol matrix, reshape -> (S,D))
              Z[:,  S*D :(S+1)*D]   = Z^1    (agent 1)
              Z[:,(S+1)*D:(S+2)*D]  = Z^2    (agent 2)
              ...

Dimension layout of Y:

    1-stock:  Y  has shape (batch, 1+I)   -- [S, Y^1, ..., Y^I]
    S-stock:  Y  has shape (batch, S+I)   -- [S^1,...,S^S, Y^1,...,Y^I]

Generator changes (Theorem 3.3.1 eq 3.3.3):

    phi0  (drift of stock equations, returns (batch, S)):
        A   = sigma^T 1  +  sum_k alpha^k Z^k       in R^d
        f^0_s = <sigma_s, A>  for s=1,...,S

    phi_i (drift of agent i equation, returns (batch, 1)):
        v   = 1*sigma + sum_k alpha^k Z^k - Z^i     in R^d
              (= sigma^T 1 + sum_k alpha_k Z^k - Z^i)
        f^i = 0.5 |Z^i|^2
              - 0.5 |v sigma^T (sigma sigma^T)^{-1} sigma|^2
              * 1_{det(sigma sigma^T) != 0}

    theta (optimal portfolio, returns list of (batch, S)):
        theta^i = 1_S  +  (sum_k alpha^k Z^k - Z^i) sigma^T (sigma sigma^T)^{-1}
                                                              (eq 3.3.4)
"""

import torch
import numpy as np
from torch import nn
import time
from abc import ABC, abstractmethod

from core.network import FeedForwardNN as myNN
from core.data_generator import BrownianMotionGenerator
from core.base_solver import FBSDEBase


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe projection norm squared
#   computes  |v  sigma^T (sigma sigma^T)^{-1} sigma|^2
#   for v: (M, d),  sigma: (M, S, d)
#   returns (M, 1)
# ─────────────────────────────────────────────────────────────────────────────
def _proj_norm_sq(v, sigma, reg=1e-8):
    """
    Compute ||v P_sigma||^2 = v sigma^T (sigma sigma^T)^{-1} sigma v^T

    Args:
        v     : (M, d)
        sigma : (M, S, d)
        reg   : regularisation added to (sigma sigma^T) diagonal

    Returns:
        (M, 1)
    """
    S = sigma.shape[1]
    device = v.device

    # w = v sigma^T  in R^{M x S}
    # sigma: (M, S, d),  v: (M, d, 1)  ->  (M, S, d) @ (M, d, 1) = (M, S, 1)
    w = (sigma @ v.unsqueeze(-1)).squeeze(-1)          # (M, S)

    # sigma sigma^T  in R^{M x S x S}
    sst = sigma @ sigma.transpose(-1, -2)              # (M, S, S)

    # regularise for numerical stability
    eye = reg * torch.eye(S, device=device).unsqueeze(0)
    sst_reg = sst + eye

    # solve (sigma sigma^T) x = w  ->  x = (sigma sigma^T)^{-1} w
    x = torch.linalg.solve(sst_reg, w.unsqueeze(-1)).squeeze(-1)   # (M, S)

    # ||v P||^2 = w^T x
    proj_sq = (w * x).sum(dim=-1, keepdim=True)       # (M, 1)
    return proj_sq


# ─────────────────────────────────────────────────────────────────────────────
# Base class  (replaces TerminalSolver for multi-stock)
# ─────────────────────────────────────────────────────────────────────────────
class TerminalSolverMultiStock(FBSDEBase, ABC):
    """
    Abstract base class for multi-stock Radner equilibrium FBSDE solvers.

    Config keys (in addition to FBSDEBase keys):
        S          : number of risky assets (M in the paper)
        drift_D    : (unused drift placeholder)
        sigD       : (S, d) tensor  -- terminal dividend volatility per stock
        sigE       : (I, d) tensor  -- endowment volatility per agent
        alpha      : list of I floats
        epsilon    : float  (reg / mask threshold)
        N_option   : float
        a_expo     : float
    """

    def __init__(self, config):
        super().__init__(config)
        self.S        = config['S']               # number of stocks
        self.drift_D  = config['drift_D']
        self.sigD     = config['sigD'].to(self.device)   # (S, d)
        self.muE      = config['muE'].to(self.device)
        self.sigE     = config['sigE'].to(self.device)   # (I, d)
        self.alpha    = config['alpha']
        self.epsilon  = config['epsilon']
        self.N_option = config['N_option']
        self.a_expo   = config['a_expo']

    # ── terminal conditions ──────────────────────────────────────────────────

    def dividend_process(self, t, X):
        """
        Terminal dividend for each stock.

        Returns:
            (M_batch, S)  -- one value per stock
        """
        # sigD: (S, d),  X: (M_batch, d)
        # row s:  xi^s = sigD[s] . X
        return X @ self.sigD.T                     # (M_batch, S)

    def dividend_gradient(self, t, X):
        """
        Gradient of dividend w.r.t. X.

        Returns:
            (M_batch, S, d)  -- one gradient row per stock
        """
        S = self.S
        M = X.shape[0]
        # sigD: (S, d) -> broadcast to (M_batch, S, d)
        return self.sigD.unsqueeze(0).expand(M, -1, -1)   # (M_batch, S, d)

    def endowment_process(self, t, X, i):
        """
        Endowment payoff for agent i (put-option style, same as 1-stock version).

        Returns:
            (M_batch, 1)
        """
        N_option = self.N_option
        a        = self.a_expo
        X2       = (self.sigE[i, :] * X)[:, 1:2]         # second Brownian, (M, 1)
        payoff   = (1 - a) * X2 + a * torch.minimum(X2, torch.zeros_like(X2))
        return -N_option * payoff if i == 0 else N_option * payoff

    def endowment_gradient(self, t, X, i):
        """
        Gradient of endowment w.r.t. X.

        Returns:
            (M_batch, d)
        """
        X = X.detach().requires_grad_(True)
        E = self.endowment_process(t, X, i)
        grad = torch.autograd.grad(
            E, X, grad_outputs=torch.ones_like(E)
        )[0]
        X = X.detach()
        return grad                                        # (M_batch, d)

    # ── state process ────────────────────────────────────────────────────────

    def mu(self, t, X):
        return super().mu(t, X)

    def sig(self, t, X):
        """Identity volatility matrix for state process X = W."""
        M_batch, d = X.shape
        return torch.eye(d, device=self.device).unsqueeze(0).expand(M_batch, -1, -1)

    # ── generators ───────────────────────────────────────────────────────────

    def _extract_sigma_and_agents(self, Z):
        """
        Split Z into sigma (stock vol matrix) and agent Z-processes.

        Z layout:  [sigma (S*D) | Z^1 (D) | Z^2 (D) | ... | Z^I (D)]

        Returns:
            sigma    : (M_batch, S, D)
            Z_agents : list of I tensors, each (M_batch, D)
        """
        S = self.S
        D = self.D
        I = self.I
        sigma    = Z[:, :S*D].reshape(-1, S, D)
        Z_agents = [Z[:, (S+i)*D:(S+i+1)*D] for i in range(I)]
        return sigma, Z_agents

    def _aggregate_A(self, sigma, Z_agents):
        """
        Compute A = sigma^T 1  +  sum_k alpha_k Z^k   in R^{M x D}

        sigma^T 1  =  sum of rows of sigma  =  sigma.sum(dim=1)
        """
        A = sigma.sum(dim=1)                              # (M, D)
        for k, Zk in enumerate(Z_agents):
            A = A + self.alpha[k] * Zk
        return A                                          # (M, D)

    def phi0(self, t, X, Z, epsilon):
        """
        Generator for the S stock equations.

        f^0_s = <sigma_s, A>   for s = 0,...,S-1

        Returns:
            (M_batch, S)
        """
        sigma, Z_agents = self._extract_sigma_and_agents(Z)
        A = self._aggregate_A(sigma, Z_agents)            # (M, D)

        # f^0_s = sigma_s . A  ->  batched dot product over D
        # sigma: (M, S, D),  A: (M, D) -> (M, D, 1)
        F0 = (sigma @ A.unsqueeze(-1)).squeeze(-1)        # (M, S)
        return F0

    def phi(self, t, X, Z, i, epsilon):
        """
        Generator for agent i (1-indexed, consistent with original code).

        f^i = 0.5 |Z^i|^2
              - 0.5 |v sigma^T (sigma sigma^T)^{-1} sigma|^2 * mask

        where v = 1*sigma + sum_k alpha_k Z^k - Z^i
                = A - Z^i  (A already contains sigma^T 1)

        Returns:
            (M_batch, 1)
        """
        sigma, Z_agents = self._extract_sigma_and_agents(Z)
        A    = self._aggregate_A(sigma, Z_agents)         # (M, D)
        Zi   = Z_agents[i - 1]                            # 1-indexed -> 0-indexed

        v    = A - Zi                                      # (M, D)

        # det mask
        sst  = sigma @ sigma.transpose(-1, -2)            # (M, S, S)
        det  = torch.linalg.det(sst)                      # (M,)
        mask = (det.abs() > epsilon).float().unsqueeze(-1) # (M, 1)

        part2 = 0.5 * (Zi ** 2).sum(dim=-1, keepdim=True) # (M, 1)
        part1 = 0.5 * _proj_norm_sq(v, sigma) * mask      # (M, 1)

        return part2 - part1

    def theta(self, Y, Z, K):
        """
        Optimal portfolio for each agent (Theorem 3.3.1 eq 3.3.4).

        theta^i = 1_S  +  (sum_k alpha_k Z^k - Z^i) sigma^T (sigma sigma^T)^{-1}

        Returns:
            list of I tensors, each (M_batch, S)
        """
        S = self.S
        D = self.D
        I = self.I
        device = Z.device

        sigma, Z_agents = self._extract_sigma_and_agents(Z)

        # sum_k alpha_k Z^k
        tmp = torch.zeros(Z.shape[0], D, device=device)
        for k, Zk in enumerate(Z_agents):
            tmp = tmp + self.alpha[k] * Zk                # (M, D)

        sst = sigma @ sigma.transpose(-1, -2)             # (M, S, S)
        det = torch.linalg.det(sst)                       # (M,)
        mask = (det.abs() > 1e-8).float()                 # (M,)

        theta_list = []
        for i in range(I):
            Zi   = Z_agents[i]                            # (M, D)
            diff = (tmp - Zi).unsqueeze(-1)               # (M, D, 1)

            # sigma @ diff: (M, S, D) @ (M, D, 1) = (M, S, 1)
            sigma_diff = sigma @ diff                      # (M, S, 1)

            # (sigma sigma^T)^{-1} sigma_diff
            reg = 1e-8 * torch.eye(S, device=device).unsqueeze(0)
            x   = torch.linalg.solve(sst + reg, sigma_diff).squeeze(-1)  # (M, S)

            ones_S   = torch.ones(Z.shape[0], S, device=device)
            theta_i  = ones_S + x * mask.unsqueeze(-1)    # (M, S)
            theta_list.append(theta_i)

        return theta_list


# ─────────────────────────────────────────────────────────────────────────────
# Solver variant 1  (single NN outputs both Y and Z)
# ─────────────────────────────────────────────────────────────────────────────
class RadnerEquilibriumSolverMulti1(TerminalSolverMultiStock):
    """
    Single neural network variant for multi-stock equilibrium.

    NN output layout:
        y  : first (S+I) neurons  ->  [S^1,...,S^S, Y^1,...,Y^I]
        z  : next  (S+I)*D neurons -> [sigma (S rows), Z^1,...,Z^I]
    """

    def __init__(self, config):
        super().__init__(config)
        self.layers = config['layers']
        self.model  = myNN(self.layers).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['learning_rate']
        )

    def net_u(self, t, x):
        """
        Returns:
            y : (M, S+I)
            z : (M, (S+I)*D)
        """
        Ii = self.S + self.I
        result = self.model(t, x)
        y = result[:, :Ii]
        z = result[:, Ii: Ii * (self.D + 1)]
        return y, z

    def loss_func(self, t, W):
        S = self.S
        I = self.I
        D = self.D
        Ii = S + I

        loss = torch.tensor([0.0], device=self.device)

        t0 = t[:, 0, :].float()
        W0 = W[:, 0, :].float()

        Y0, Z0 = self.net_u(t0, W0)
        # apply sig transformation to each block of Z
        for i in range(Ii):
            Z0[:, i*D:(i+1)*D] = (
                Z0[:, i*D:(i+1)*D].unsqueeze(1) @ self.sig(t0, W0)
            ).squeeze(1)

        for time in range(1, self.N + 1):
            t1 = t[:, time, :].float()
            W1 = W[:, time, :].float()
            dW = W1 - W0                                   # (M, D)

            # ── build Y1_pred ──────────────────────────────────────────────
            Y1_temp = []

            # --- S stock equations ---
            # phi0 returns (M, S);  diffusion: sigma_s . dW  for each s
            sigma_block = Z0[:, :S*D].reshape(-1, S, D)   # (M, S, D)
            F0_val      = self.phi0(t0, W0, Z0, self.epsilon)  # (M, S)
            stoch_stock = (sigma_block @ dW.unsqueeze(-1)).squeeze(-1)  # (M, S)
            Y_stocks_pred = Y0[:, :S] + F0_val * (t1 - t0) + stoch_stock  # (M, S)
            Y1_temp.append(Y_stocks_pred)

            # --- I agent equations ---
            for j in range(I):
                Fj_val = self.phi(t0, W0, Z0, j + 1, self.epsilon)    # (M, 1)
                Zj     = Z0[:, (S+j)*D:(S+j+1)*D]                     # (M, D)
                stoch_agent = (Zj * dW).sum(dim=-1, keepdim=True)     # (M, 1)
                pred = Y0[:, S+j].unsqueeze(1) + Fj_val * (t1 - t0) + stoch_agent
                Y1_temp.append(pred)

            Y1_pred = torch.cat(Y1_temp, dim=1)            # (M, S+I)

            # ── get Y1, Z1 from network ─────────────────────────────────────
            Y1, Z1 = self.net_u(t1, W1)
            for i in range(Ii):
                Z1[:, i*D:(i+1)*D] = (
                    Z1[:, i*D:(i+1)*D].unsqueeze(1) @ self.sig(t1, W1)
                ).squeeze(1)

            loss += nn.MSELoss(reduction='mean')(Y1, Y1_pred)
            t0, W0, Y0, Z0 = t1, W1, Y1, Z1

        # ── terminal loss ───────────────────────────────────────────────────
        D_T = self.dividend_process(t1, W1)                # (M, S)
        E_T = [self.endowment_process(t1, W1, i)
               for i in range(I)]                          # list of (M, 1)

        Y_terminal = torch.cat([D_T] + E_T, dim=1).detach()   # (M, S+I)
        loss += nn.MSELoss(reduction='mean')(Y1, Y_terminal)

        return loss

    def train(self, NIter, epoch, patience, min_delta=1e-6):
        start_time = time.perf_counter()
        best_val_loss = float("inf")
        best_epoch    = 0
        no_improve    = 0

        for j in range(epoch):
            print(f"\nEpoch {j+1}\n" + "-"*30)
            for i in range(NIter):
                tb, Wb = self.t_train[i*self.M:(i+1)*self.M], \
                         self.W_train[i*self.M:(i+1)*self.M]
                loss = self.loss_func(tb, Wb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.history['train_loss'].append(loss.item())
                if (i+1) % 200 == 0:
                    print(f"  iter {i+1}/{NIter}  loss={loss.item():.3e}  "
                          f"t={time.perf_counter()-start_time:.1f}s")

            val_loss = self.loss_func(self.t_valid, self.W_valid).item()
            self.history['valid_loss'].append(val_loss)
            print(f"  val_loss={val_loss:.3e}")

            if best_val_loss - val_loss > min_delta:
                best_val_loss = val_loss
                best_epoch    = j
                no_improve    = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  -> saved checkpoint")
            else:
                no_improve += 1
                print(f"  no improvement {no_improve}/{patience}")

        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.history.update(dict(
            train_time    = time.perf_counter() - start_time,
            best_val_loss = best_val_loss,
            best_epoch    = best_epoch,
        ))

    def predict(self, t, W):
        S = self.S
        I = self.I
        D = self.D

        t0 = t[:, 0, :].float()
        W0 = W[:, 0, :].float()

        Y0, Z0  = self.net_u(t0, W0)
        Y_path  = [[] for _ in range(S + I)]
        Th_path = [[] for _ in range(I)]

        for idx in range(S + I):
            Y_path[idx].append(Y0[:, idx].unsqueeze(1))
        for th in self.theta(Y0, Z0, self.K):
            Th_path[self.theta(Y0, Z0, self.K).index(th)].append(th.unsqueeze(1)
                if th.dim()==2 else th)

        # simpler theta loop
        th0 = self.theta(Y0, Z0, self.K)
        Th_path = [[th0[i]] for i in range(I)]

        for time in range(1, self.N + 1):
            t1 = t[:, time, :].float()
            W1 = W[:, time, :].float()
            Y1, Z1 = self.net_u(t1, W1)
            for idx in range(S + I):
                Y_path[idx].append(Y1[:, idx].unsqueeze(1))
            th1 = self.theta(Y1, Z1, self.K)
            for i in range(I):
                Th_path[i].append(th1[i])
            Y0, t0, W0 = Y1, t1, W1

        for idx in range(S + I):
            Y_path[idx] = torch.stack(Y_path[idx], dim=1)   # (K, N+1, 1)
        for i in range(I):
            Th_path[i]  = torch.stack(Th_path[i],  dim=1)   # (K, N+1, S)

        D_T     = self.dividend_process(t1, W1)
        E_T     = [self.endowment_process(t1, W1, i) for i in range(I)]
        Y_term  = torch.cat([D_T] + E_T, dim=1).detach()
        print(f"Terminal loss: {nn.MSELoss()(Y1, Y_term).item():.4f}")

        return Y_path, Th_path


# ─────────────────────────────────────────────────────────────────────────────
# Solver variant 4  (separate NN for Y and Z)
# ─────────────────────────────────────────────────────────────────────────────
class RadnerEquilibriumSolverMulti4(TerminalSolverMultiStock):
    """
    Two-network variant: model_y predicts Y, model_z predicts Z.

    model_y output: (S+I)
    model_z output: (S+I)*D
    """

    def __init__(self, config):
        super().__init__(config)
        self.layers_y = config['layers_y']
        self.layers_z = config['layers_z']
        self.model_y  = myNN(self.layers_y).to(self.device)
        self.model_z  = myNN(self.layers_z).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model_y.parameters()) + list(self.model_z.parameters()),
            lr=self.learning_rate
        )

    def net_y(self, t, x):
        return self.model_y(t, x)          # (M, S+I)

    def net_z(self, t, x):
        return self.model_z(t, x)          # (M, (S+I)*D)

    def loss_func(self, t, W):
        S  = self.S
        I  = self.I
        D  = self.D
        Ii = S + I

        loss = torch.tensor([0.0], device=self.device)

        t0 = t[:, 0, :].float()
        W0 = W[:, 0, :].float()

        Y0 = self.net_y(t0, W0)
        Z0 = self.net_z(t0, W0)
        for i in range(Ii):
            Z0[:, i*D:(i+1)*D] = (
                Z0[:, i*D:(i+1)*D].unsqueeze(1) @ self.sig(t0, W0)
            ).squeeze(1)

        for time in range(1, self.N + 1):
            t1 = t[:, time, :].float()
            W1 = W[:, time, :].float()
            dW = W1 - W0

            # ── stock equations ─────────────────────────────────────────────
            sigma_block  = Z0[:, :S*D].reshape(-1, S, D)
            F0_val       = self.phi0(t0, W0, Z0, self.epsilon)   # (M, S)
            stoch_stock  = (sigma_block @ dW.unsqueeze(-1)).squeeze(-1)
            Y_stocks_pred = Y0[:, :S] + F0_val * (t1 - t0) + stoch_stock

            # ── agent equations ─────────────────────────────────────────────
            Y_agents_pred = []
            for j in range(I):
                Fj  = self.phi(t0, W0, Z0, j+1, self.epsilon)    # (M, 1)
                Zj  = Z0[:, (S+j)*D:(S+j+1)*D]
                stoch = (Zj * dW).sum(-1, keepdim=True)
                Y_agents_pred.append(
                    Y0[:, S+j].unsqueeze(1) + Fj * (t1 - t0) + stoch
                )

            Y1 = torch.cat([Y_stocks_pred] + Y_agents_pred, dim=1)

            Z1 = self.net_z(t1, W1)
            for i in range(Ii):
                Z1[:, i*D:(i+1)*D] = (
                    Z1[:, i*D:(i+1)*D].unsqueeze(1) @ self.sig(t1, W1)
                ).squeeze(1)

            
            t0, W0, Y0, Z0 = t1, W1, Y1, Z1

        # ── terminal loss ───────────────────────────────────────────────────
        D_T = self.dividend_process(t1, W1)
        E_T = [self.endowment_process(t1, W1, i) for i in range(I)]
        Y_term = torch.cat([D_T] + E_T, dim=1).detach()
        loss  += nn.MSELoss(reduction='mean')(Y1, Y_term)

        return loss

    def train(self, NIter, epoch, patience, min_delta=1e-6):
        start_time    = time.perf_counter()
        best_val_loss = float("inf")
        best_epoch    = 0

        for j in range(epoch):
            print(f"\nEpoch {j+1}\n" + "-"*30)
            for i in range(NIter):
                tb = self.t_train[i*self.M:(i+1)*self.M]
                Wb = self.W_train[i*self.M:(i+1)*self.M]
                loss = self.loss_func(tb, Wb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.history['train_loss'].append(loss.item())
                if (i+1) % 200 == 0:
                    print(f"  iter {i+1}/{NIter}  loss={loss.item():.3e}")

            val_loss = self.loss_func(self.t_valid, self.W_valid).item()
            self.history['valid_loss'].append(val_loss)
            print(f"  val_loss={val_loss:.3e}")

            if best_val_loss - val_loss > min_delta:
                best_val_loss = val_loss
                best_epoch    = j
                torch.save({
                    "model_y": self.model_y.state_dict(),
                    "model_z": self.model_z.state_dict(),
                }, self.checkpoint_path)
                print(f"  -> saved checkpoint")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model_y.load_state_dict(ckpt["model_y"])
        self.model_z.load_state_dict(ckpt["model_z"])
        self.history.update(dict(
            train_time    = time.perf_counter() - start_time,
            best_val_loss = best_val_loss,
            best_epoch    = best_epoch,
        ))

    def predict(self, t, W):
        S = self.S
        I = self.I
        D = self.D
        Ii = S + I

        t0 = t[:, 0, :].float()
        W0 = W[:, 0, :].float()

        Y0 = self.net_y(t0, W0)
        Z0 = self.net_z(t0, W0)
        for i in range(Ii):
            Z0[:, i*D:(i+1)*D] = (
                Z0[:, i*D:(i+1)*D].unsqueeze(1) @ self.sig(t0, W0)
            ).squeeze(1)

        Y_path  = [[Y0[:, idx].unsqueeze(1)] for idx in range(S+I)]
        Z_path  = [[Z0[:, idx*D:(idx+1)*D].unsqueeze(1)] for idx in range(S+I)]
        Th_path = [[th] for th in self.theta(Y0, Z0, self.K)]

        for time in range(1, self.N + 1):
            t1 = t[:, time, :].float()
            W1 = W[:, time, :].float()
            dW = W1 - W0

            Z1 = self.net_z(t1, W1)
            # 获取下一时刻的 Z1，并 ！！修复！！ 乘上波动率 sig
            Z1 = self.net_z(t1, W1)
            for i in range(Ii):
                Z1[:, i*D:(i+1)*D] = (
                    Z1[:, i*D:(i+1)*D].unsqueeze(1) @ self.sig(t1, W1)
                ).squeeze(1)

            sigma_block  = Z0[:, :S*D].reshape(-1, S, D)
            F0_val       = self.phi0(t0, W0, Z0, self.epsilon)
            stoch_stock  = (sigma_block @ dW.unsqueeze(-1)).squeeze(-1)
            Y_stocks_new = Y0[:, :S] + F0_val*(t1-t0) + stoch_stock

            Y_agents_new = []
            for j in range(I):
                Fj   = self.phi(t0, W0, Z0, j+1, self.epsilon)
                Zj   = Z0[:, (S+j)*D:(S+j+1)*D]
                stch = (Zj * dW).sum(-1, keepdim=True)
                Y_agents_new.append(Y0[:, S+j].unsqueeze(1) + Fj*(t1-t0) + stch)

            Y1 = torch.cat([Y_stocks_new] + Y_agents_new, dim=1)

            for idx in range(S+I):
                Y_path[idx].append(Y1[:, idx].unsqueeze(1))
                Z_path[idx].append(Z1[:, idx*D:(idx+1)*D].unsqueeze(1))
            for i, th in enumerate(self.theta(Y1, Z1, self.K)):
                Th_path[i].append(th)

            Y0, Z0, t0, W0 = Y1, Z1, t1, W1

        for idx in range(S+I):
            Y_path[idx] = torch.stack(Y_path[idx], dim=1)   # (K, N+1, 1)
            Z_path[idx] = torch.stack(Z_path[idx], dim=1)   # (K, N+1, D)
            
        for i in range(I):
            Th_path[i]  = torch.stack(Th_path[i],  dim=1)   # (K, N+1, S)

        D_T    = self.dividend_process(t1, W1)
        E_T    = [self.endowment_process(t1, W1, i) for i in range(I)]
        Y_term = torch.cat([D_T] + E_T, dim=1).detach()
        print(f"Terminal loss: {nn.MSELoss()(Y1, Y_term).item():.4f}")

        return Y_path, Z_path, Th_path