import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")


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
        axes[0].plot([], [], color=self.colors["learned"], label="Learned")
        axes[0].plot([], [], color=self.colors["exact"], linestyle="-.", label="Exact")
        axes[0].legend(fontsize=10, loc="upper left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig

    # =====================================================
    # Figure 2: Relative error (mean ± 2 std)
    # =====================================================
    def plot_relative_errors(
        self,
        Y_pred,
        Y_exact,
        zeta=1e-6,
        save_path=None,
        max_cols=2,
    ):
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

            error = np.sqrt(
                (Y_pred[p] - Y_exact[p]) ** 2
                / np.maximum(zeta**2, Y_exact[p] ** 2)
            )

            mean_err = error.mean(axis=0)
            std_err = error.std(axis=0)

            ax.plot(mean_err, color=self.colors["mean"], label="mean")
            ax.plot(
                mean_err + 2 * std_err,
                color=self.colors["std"],
                linestyle="--",
                linewidth=2,
                label="mean + 2 std",
            )

            ax.set_title(titles[p], fontsize=18)

            ax.set_xticks(np.linspace(0, mean_err.shape[0] - 1, 11, dtype=int))
            ax.set_xticklabels(
                np.round(np.linspace(0.0, 1.0, 11), 1)
            )

            if p % max_cols == 0:
                ax.set_ylabel("relative error", fontsize=14)
            if p >= num_proc - max_cols:
                ax.set_xlabel("t", fontsize=14)

            ax.legend(fontsize=10, loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        return fig
