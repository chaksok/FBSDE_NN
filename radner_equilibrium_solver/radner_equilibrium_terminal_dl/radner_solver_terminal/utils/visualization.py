import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")


# def plot_training_loss(history, save_path=None):
#     train_loss = np.array(history['train_error'])

#     plt.figure(figsize=(8, 5))
#     plt.plot(train_loss, linewidth=1.5)
#     plt.yscale("log")
#     plt.xlabel("Iteration")
#     plt.ylabel("Training error (log scale)")
#     plt.title("Training error decay")
#     plt.grid(alpha=0.3)
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#     plt.show()


def plot_training_loss_and_error(history, save_path=None):

    train_loss = np.array(history['train_loss'])
    train_error = np.array(history['train_error'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ---- loss ----
    axes[0].plot(train_loss, linewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Training loss (log scale)")
    axes[0].set_title("Loss decay")
    axes[0].grid(alpha=0.3)

    # ---- error ----
    axes[1].plot(train_error, linewidth=1.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Training error (log scale)")
    axes[1].set_title("Error decay")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    # plt.show()



# def plot_loss_decay_MN(loss_records, save_path=None):
#     plt.figure(figsize=(10, 6))

#     for (M_, N_), hist in loss_records.items():
#         loss = np.array(hist['train_error'])
#         plt.plot(
#             loss,
#             linewidth=2,
#             label=f"M={M_}, N={N_}",
#         )

#     plt.yscale("log")
#     plt.xlabel("Iteration")
#     plt.ylabel("Training error (log scale)")
#     plt.title("Training error decay for different (M, N)")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight")
#     plt.show()


def plot_loss_decay_MN(loss_records, save_path=None):

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ======================
    # Left: training loss
    # ======================
    for (M_, N_), hist in loss_records.items():
        loss = np.array(hist['train_loss'])
        axes[0].plot(
            loss,
            linewidth=2,
            label=f"M={M_}, N={N_}",
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Training loss (log scale)")
    axes[0].set_title("Loss decay for different $(M,N)$")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # ======================
    # middle: validation L2 error
    # ======================
    for (M_, N_), hist in loss_records.items():
        error = np.array(hist['valid_error'])
        axes[1].plot(
            error,
            linewidth=2,
            label=f"M={M_}, N={N_}",
        )

    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Validation error (log scale)")
    axes[1].set_title("L2 Error decay for different $(M,N)$")
    axes[1].grid(alpha=0.3)
    axes[1].legend()


    # ======================
    # middle: Validation Y0 error
    # ======================
    for (M_, N_), hist in loss_records.items():
        error = np.array(hist['Y0_error'])
        axes[2].plot(
            error,
            linewidth=2,
            label=f"M={M_}, N={N_}",
        )

    axes[2].set_yscale("log")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Validation error (log scale)")
    axes[2].set_title("Y0 Error decay for different $(M,N)$")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    # plt.show()

class RadnerVisualizer:
    """
    Visualization tools for Radner equilibrium results.
    Fully generalized to arbitrary number of agents.
    """

    def __init__(self):
        self.colors = {
            "learned": "#2A77AC",
            "exact": "#C9244C",
            "mean": "#1B7C3D",
            "std": "#F16C23",
        }

  
    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    @staticmethod
    def _subplot_grid(n, max_cols=2):
        cols = min(max_cols, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(8 * cols, 4 * rows), squeeze=False
        )
        return fig, axes.flatten()

    # =====================================================
    # Figure 1: Path comparison (learned vs exact)
    # =====================================================
    def plot_paths(
        self,
        Y_pred,
        Y_exact,
        save_path=None,
        max_cols=2,
    ):
        """
        Parameters
        ----------
        Y_pred : list of arrays/tensors
            Each element shape (num_paths, T)
        Y_exact : list of arrays/tensors
            Same structure as Y_pred
        """

        Y_pred = [self._to_numpy(y).squeeze() for y in Y_pred]
        Y_exact = [self._to_numpy(y).squeeze() for y in Y_exact]

        num_proc = len(Y_pred)
        titles = ["Price Process of Risky Asset S"] + [
            f"Certainty equivalent of agent {i}"
            for i in range(1, num_proc)
        ]

        fig, axes = self._subplot_grid(num_proc, max_cols)

        for p in range(num_proc):
            ax = axes[p]
            Yp = Y_pred[p]
            Ye = Y_exact[p]

            for k in range(5):  # plot 5 sample paths
                ax.plot(
                    Yp[k],
                    color=self.colors["learned"],
                    linewidth=2,
                )
                ax.plot(
                    Ye[k],
                    color=self.colors["exact"],
                    linestyle="-.",
                    linewidth=2,
                )

                # terminal & initial markers
                ax.scatter(
                    len(Yp[k]) - 1,
                    Yp[k, -1],
                    marker="s",
                    s=60,
                    color="black",
                )
                ax.scatter(
                    0,
                    Yp[k, 0],
                    marker="o",
                    s=80,
                    color="black",
                )

            ax.set_title(titles[p], fontsize=18)

            ax.set_xticks(np.linspace(0, Yp.shape[1] - 1, 11, dtype=int))
            ax.set_xticklabels(
                np.round(np.linspace(0.0, 1.0, 11), 1)
            )

            if p % max_cols == 0:
                ax.set_ylabel("Value", fontsize=14)
            if p >= num_proc - max_cols:
                ax.set_xlabel("t", fontsize=14)

        # legend once
        for i in range(num_proc):
            axes[i].plot([], [], color=self.colors["learned"], label="Learned")
            axes[i].plot([], [], color=self.colors["exact"], linestyle="-.", label="Exact")
            axes[i].legend(fontsize=10, loc="upper left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig



    def plot_price_and_theta(
        self,
        Y_pred,
        Y_exact,
        Theta_pred,
        Theta_real,
        save_path=None,
        max_cols=2,
    ):
        """
        Subplot 0 : price process S
        Subplot >=1 : theta comparison for each agent

        Parameters
        ----------
        Y_pred, Y_exact : list of tensors
            Y_pred[0] is price process
        Theta_pred : list of tensors
            Theta_pred[i] shape (K, T, 1)
        Theta_real : list of tensors
            Theta_real[i] shape (T, 1) or (1, T, 1)
        """

        # ---------- to numpy ----------
        Y_pred = [self._to_numpy(y).squeeze() for y in Y_pred]
        Y_exact = [self._to_numpy(y).squeeze() for y in Y_exact]

        Theta_pred = [self._to_numpy(t).squeeze() for t in Theta_pred]
        Theta_real = [self._to_numpy(t) for t in Theta_real]

        I = len(Theta_pred)          # number of agents
        num_proc = I + 1             # price + agents

        titles = ["Price Process of Risky Asset S"] + [
            f"Portfolio strategy of agent {i}"
            for i in range(1, num_proc)
        ]

        fig, axes = self._subplot_grid(num_proc, max_cols)

        for p in range(num_proc):
            ax = axes[p]

            # ==================================================
            # p = 0 : PRICE PROCESS (unchanged)
            # ==================================================
            if p == 0:
                Yp = Y_pred[0]
                Ye = Y_exact[0]

                for k in range(min(5, Yp.shape[0])):
                    ax.plot(
                        Yp[k],
                        color=self.colors["learned"],
                        linewidth=2,
                    )
                    ax.plot(
                        Ye[k],
                        color=self.colors["exact"],
                        linestyle="-.",
                        linewidth=2,
                    )

                    ax.scatter(
                        len(Yp[k]) - 1,
                        Yp[k, -1],
                        marker="s",
                        s=60,
                        color="black",
                    )
                    ax.scatter(
                        0,
                        Yp[k, 0],
                        marker="o",
                        s=80,
                        color="black",
                    )

            # ==================================================
            # p >= 1 : THETA COMPARISON
            # ==================================================
            else:
                Tp = Theta_pred[p - 1]     # (K, T)
                Tr = Theta_real[p - 1]     # (T,) or (1,T)

                if Tr.ndim == 2:
                    Tr = Tr[0]

                # learned: multiple paths
                for k in range(min(5, Tp.shape[0])):
                    ax.plot(
                        Tp[k],
                        color=self.colors["learned"],
                        linewidth=2,
                    )

                # exact: single path
                ax.plot(
                    Tr,
                    color=self.colors["exact"],
                    linestyle="-.",
                    linewidth=3,
                )

                theta_min = Tr.min()
                theta_max = Tr.max()

                ax.set_ylim(
                    theta_min - 0.1,
                    theta_max + 0.1,
            )

            # ---------- formatting ----------
            ax.set_title(titles[p], fontsize=18)

            ax.set_xticks(np.linspace(0, Y_pred[0].shape[1] - 1, 11, dtype=int))
            ax.set_xticklabels(np.round(np.linspace(0.0, 1.0, 11), 1))

            if p % max_cols == 0:
                ax.set_ylabel("Value", fontsize=14)
            if p >= num_proc - max_cols:
                ax.set_xlabel("t", fontsize=14)

        # ---------- legend once ----------
        for i in range(num_proc):
            axes[i].plot([], [], color=self.colors["learned"], label="Learned")
            axes[i].plot([], [], color=self.colors["exact"], linestyle="-.", label="Exact")
            axes[i].legend(fontsize=10, loc="upper left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig
    

    def plot_price_and_theta_pred(
        self,
        Y_pred,
        Theta_pred,
        save_path=None,
        max_cols=2,
    ):
        """
        Subplot 0 : price process S
        Subplot >=1 : theta comparison for each agent

        Parameters
        ----------
        Y_pred, Y_exact : list of tensors
            Y_pred[0] is price process
        Theta_pred : list of tensors
            Theta_pred[i] shape (K, T, 1)
        Theta_real : list of tensors
            Theta_real[i] shape (T, 1) or (1, T, 1)
        """

        # ---------- to numpy ----------
        Y_pred = [self._to_numpy(y).squeeze() for y in Y_pred]
       
        Theta_pred = [self._to_numpy(t).squeeze() for t in Theta_pred]
       

        I = len(Theta_pred)          # number of agents
        num_proc = I + 1             # price + agents

        titles = ["Price Process of Risky Asset S"] + [
            f"Portfolio strategy of agent {i}"
            for i in range(1, num_proc)
        ]

        fig, axes = self._subplot_grid(num_proc, max_cols)

        for p in range(num_proc):
            ax = axes[p]

            # ==================================================
            # p = 0 : PRICE PROCESS (unchanged)
            # ==================================================
            if p == 0:
                Yp = Y_pred[0]
               

                for k in range(min(5, Yp.shape[0])):
                    ax.plot(
                        Yp[k],
                        color=self.colors["learned"],
                        linewidth=2,
                    )

                    ax.scatter(
                        len(Yp[k]) - 1,
                        Yp[k, -1],
                        marker="s",
                        s=60,
                        color="black",
                    )
                    ax.scatter(
                        0,
                        Yp[k, 0],
                        marker="o",
                        s=80,
                        color="black",
                    )

            # ==================================================
            # p >= 1 : THETA COMPARISON
            # ==================================================
            else:
                Tp = Theta_pred[p - 1]     # (K, T)
                



                # learned: multiple paths
                for k in range(min(5, Tp.shape[0])):
                    ax.plot(
                        Tp[k],
                        color=self.colors["learned"],
                        linewidth=2,
                    )

            

            # ---------- formatting ----------
            ax.set_title(titles[p], fontsize=18)

            ax.set_xticks(np.linspace(0, Y_pred[0].shape[1] - 1, 11, dtype=int))
            ax.set_xticklabels(np.round(np.linspace(0.0, 1.0, 11), 1))

            if p % max_cols == 0:
                ax.set_ylabel("Value", fontsize=14)
            if p >= num_proc - max_cols:
                ax.set_xlabel("t", fontsize=14)

        # ---------- legend once ----------
        for i in range(num_proc):
            axes[i].plot([], [], color=self.colors["learned"], label="Learned")
            axes[i].plot([], [], color=self.colors["exact"], linestyle="-.", label="Exact")
            axes[i].legend(fontsize=10, loc="upper left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig
    

    # =====================================================
    # Figure 2: RMSE (mean ± 2 std over paths)
    # =====================================================
    def plot_rmse_errors(
        self,
        Y_pred,
        Y_exact,
        save_path=None,
        max_cols=2,
    ):
        """
        Plot RMSE (root mean squared error) for each process.
        """
        Y_pred = [self._to_numpy(y).squeeze() for y in Y_pred]
        Y_exact = [self._to_numpy(y).squeeze() for y in Y_exact]

        num_proc = len(Y_pred)
        titles = ["Price Process of Risky Asset S"] + [
            f"Certainty equivalent of agent {i}" for i in range(1, num_proc)
        ]

        fig, axes = self._subplot_grid(num_proc, max_cols)

        for p in range(num_proc):
            ax = axes[p]

            # RMSE across paths at each time step
            error = (Y_pred[p] - Y_exact[p]) ** 2
            mean_rmse = np.sqrt(error.mean(axis=0))
            std_rmse  = np.sqrt(error.std(axis=0))  # optional: std of RMSE over paths

            # plot mean RMSE
            ax.plot(mean_rmse, color=self.colors["mean"], label="mean RMSE")

            # plot mean ± 2 std RMSE
            ax.fill_between(
                np.arange(mean_rmse.shape[0]),
                np.maximum(mean_rmse - 2*std_rmse, 0),
                mean_rmse + 2*std_rmse,
                color=self.colors["std"],
                alpha=0.3,
                label="±2 std",
            )

            ax.set_title(titles[p], fontsize=18)
            ax.set_xticks(np.linspace(0, mean_rmse.shape[0] - 1, 11, dtype=int))
            ax.set_xticklabels(np.round(np.linspace(0.0, 1.0, 11), 1))

            if p % max_cols == 0:
                ax.set_ylabel("RMSE", fontsize=14)
            if p >= num_proc - max_cols:
                ax.set_xlabel("t", fontsize=14)

            ax.legend(fontsize=10, loc="upper right")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig


    def plot_relative_errors_theta(
        self,
        Theta_pred,
        Theta_real,
        eps=1e-12,
        save_path=None,
        max_cols=2,
    ):
        """
        Plot relative errors for portfolio strategies theta^i.

        Parameters
        ----------
        Theta_pred : list of tensors/arrays
            Theta_pred[i] shape (M, T), learned portfolio strategies
        Theta_real : list of tensors/arrays
            Theta_real[i] shape (T,) or (1,T), analytical strategies
        eps : float
            Small value to avoid division by zero
        save_path : str
            Path to save the figure
        max_cols : int
            Maximum number of columns in subplot
        """

        # ---------- convert to numpy ----------
        Theta_pred = [self._to_numpy(t).squeeze() for t in Theta_pred]
        Theta_real = [self._to_numpy(t).squeeze() for t in Theta_real]

        I = len(Theta_pred)
        titles = [f"Portfolio strategy $\\theta^{i}$" for i in range(1, I + 1)]

        fig, axes = self._subplot_grid(I, max_cols)

        # =====================================================
        # Plot relative error for each agent
        # =====================================================
        for i in range(I):
            ax = axes[i]

            Tp = Theta_pred[i]        # (M, T)
            Tr = Theta_real[i]        # (T,)

            # broadcast Tr to (M,T)
            Tr = np.broadcast_to(Tr, Tp.shape)

            # relative error
            err_theta = np.abs(Tp - Tr) / (np.abs(Tr) + eps)
            mean_err = err_theta.mean(axis=0)
            std_err  = err_theta.std(axis=0)

            # plot mean ± 2 std
            ax.semilogy(mean_err, color=self.colors["mean"], label="mean")
            ax.fill_between(
                np.arange(mean_err.shape[0]),
                np.maximum(mean_err - 2*std_err, eps),
                mean_err + 2*std_err,
                color=self.colors["std"],
                alpha=0.3,
                label="±2 std",
            )

            ax.set_title(titles[i], fontsize=16)
            ax.set_xlabel("t")
            ax.set_ylabel("relative error")
            ax.legend(loc="upper right")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig


        