# /examples/oneDCorr/gen_data.py
"""
a: known input
random_field (unobserved at inference): latent
w: observation
goal: learn p(w | a)
"""
import random
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from floral.solver import OneDCorr
from scipy.stats import pearsonr, gaussian_kde
from floral.utils import sample_grf_rbf

plt.style.use("../../scripts/journal.mplstyle")

SEED = 42


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(
        prog="onedcorr",
        description="generate synthetic data for onedcorr",
    )
    parser.add_argument(
        "-n",
        "--n_unique_conditions",
        type=int,
        default=100,
        help="Number of unique input conditions",
    )

    parser.add_argument(
        "--n_fields_per_sample",
        type=int,
        default=50,
        help="Number of random field realizations per input condition",
    )

    parser.add_argument(
        "-res",
        "--resolution",
        type=int,
        default=128,
        help="Resolution of the grid",
    )
    parser.add_argument(
        "--k_range",
        type=list,
        default=[10, 14],
        help="Range of k values to sample from",
    )
    parser.add_argument(
        "--rbf_length_scale",
        type=float,
        default=0.2,
        help="Lenght scale for the RBF kernel",
    )
    parser.add_argument(
        "--rbf_sigma",
        type=float,
        default=0.25,
        help="Amplitude for the RBF kernel",
    )
    parser.add_argument(
        "--rbf_kl_modes",
        type=int,
        default=10,
        help="Number of KL modes to retain",
    )

    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    print("==" * 3 + "onedcorr" + "==" * 3)
    print(f"Number of unique conditions: {args.n_unique_conditions}")
    print(f"Number of random fields/sample: {args.n_fields_per_sample}")
    print(
        f"Total generated samples:  {args.n_unique_conditions*args.n_fields_per_sample}"
    )
    print(f"Resolution: {args.resolution}")
    print("==" * 3 + "========" + "==" * 3)

    return args


def seed_everything():
    """seed everything"""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


class MultiFidelity:
    def __init__(
        self,
        resolution,
        n_unique_conditions,
        rbf_length_scale,
        rbf_sigma,
        rbf_kl_modes,
        n_fields_per_sample: int = 1,
        plot: bool = False,
        alpha: float = 0.1,
        beta: float = 0.1,
    ):
        self.resolution = resolution
        self.n_unique_conditions = n_unique_conditions
        self.rbf_length_scale = rbf_length_scale
        self.rbf_sigma = rbf_sigma
        self.rbf_kl_modes = rbf_kl_modes
        self.n_fields_per_sample = n_fields_per_sample
        self.plot = plot
        self.alpha = alpha
        self.beta = beta
        self.gamma = 0.05  # additive independent noise scale

        self.solver_LF = OneDCorr(
            resolution=self.resolution, n_samples=n_unique_conditions
        )

        self.domain = self.solver_LF.domain

    def _get_input_conditions(self):
        """same HF and LF conditions
        tiling: [a0, a1, a2, a0, a1, a2, ....]
        """
        # sample n_unique_conditions
        a_LF = self.solver_LF._sample_input_function()
        # tile n_fields_per_sample times
        a_LF = a_LF.repeat(self.n_fields_per_sample, 1)
        return a_LF, a_LF

    def _plot_random_field(self, random_field: np.ndarray, n_plot: int = 5):
        """plot a handful of GRF realizations"""
        assert random_field.ndim == 2, "random_field must be (n_samples, N)"
        n_plot = min(n_plot, len(random_field))
        fig, ax = plt.subplots(figsize=(6, 3), layout="compressed")
        for ii in range(n_plot):
            ax.plot(self.domain.ravel(), random_field[ii], alpha=0.7, lw=1)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$z(x)$")
        ax.set_title("GRF samples (RBF kernel)")
        plt.savefig("random_field.png", dpi=150)
        plt.close()

    def _get_random_field(self, seed: int = SEED):
        coords = np.array(self.domain).T  # (N,1)
        total = self.n_unique_conditions * self.n_fields_per_sample
        random_field, _ = sample_grf_rbf(
            coords=coords,
            n_samples=total,
            length_scale=self.rbf_length_scale,
            sigma=self.rbf_sigma,
            nugget=1e-6,
            rng=np.random.default_rng(seed),
            num_modes=self.rbf_kl_modes,
        )
        return torch.tensor(random_field, dtype=torch.float32)

    def _apply_multiplicative_random_field(
        self, deterministic_field: torch.Tensor, random_field: torch.Tensor
    ):
        """Apply latent output-space stochasticity."""
        assert (
            deterministic_field.shape == random_field.shape
        ), "deterministic field and random field must have the same shape"
        return deterministic_field * (1.0 + random_field)

    def _make_sample_comparison_plot(self, u_LF, u_HF, n_conditions: int = 4):
        """Plot solution variability for the same input condition.

        Assumes data is in tile order (pre-shuffle):
        rows [i, i+ns, i+2*ns, ...] all share input condition i.
        For each selected condition, overlays mean ± std bands for both fidelities.
        """
        nf = self.n_fields_per_sample
        ns = self.n_unique_conditions
        n_conditions = min(n_conditions, ns)
        x = self.domain.ravel()

        fig, axs = plt.subplots(
            n_conditions,
            1,
            figsize=(5, n_conditions * 2.5),
            sharex=True,
            layout="compressed",
        )
        if n_conditions == 1:
            axs = [axs]

        for ii, ax in enumerate(axs):
            # tile layout: condition ii lives at rows ii, ii+ns, ii+2*ns, ...
            hf_block = u_HF[ii::ns][:nf]  # (nf, N)
            lf_block = u_LF[ii::ns][:nf]  # (nf, N)

            hf_mean = hf_block.mean(dim=0)
            hf_std = hf_block.std(dim=0)
            lf_mean = lf_block.mean(dim=0)
            lf_std = lf_block.std(dim=0)

            # LF mean ± std band (gray) + HF mean ± std band (black)
            ax.fill_between(
                x, lf_mean - lf_std, lf_mean + lf_std, color="red", alpha=0.2
            )
            ax.plot(
                x, lf_mean, color="red", alpha=0.8, lw=1.5, label="Low-fidelity mean"
            )

            ax.fill_between(x, hf_mean - hf_std, hf_mean + hf_std, color="k", alpha=0.2)
            ax.plot(x, hf_mean, color="k", lw=1.5, label=r"High-fidelity mean")
            ax.set_ylabel(r"$w(x)$")
            if ii == 0:
                ax.legend(loc="upper right")

        axs[-1].set_xlabel(r"$x$")
        plt.savefig("data_snapshot.png", dpi=150)
        plt.close()

    def _make_marginal_plot(self, u_LF, u_HF, percentage_plot: float = 0.5):
        n_plot = max(1, int(len(u_LF) * percentage_plot))
        print(f"Making marginal distribution using {n_plot}/{len(u_LF)} samples")
        # extract field
        flat_LF = u_LF[:n_plot].flatten().numpy()
        flat_HF = u_HF[:n_plot].flatten().numpy()
        flat_residual = flat_HF - flat_LF

        fig, axs = plt.subplots(1, 2, figsize=(6, 3), layout="compressed")
        axs[0].hist(flat_LF, density=True, bins=100, label=r"Low-fidelity")
        axs[0].hist(flat_HF, density=True, bins=100, label=r"High-fidelity")
        axs[0].legend()
        axs[0].set_title("State marginal")
        axs[1].hist(flat_residual, density=True, bins=100, label=r"Residual")
        axs[1].set_title("Residual marginal")

        plt.savefig("data_marginal.png")

    def _make_joint_plot(
        self,
        samples_X,
        samples_Y,
        xlabel=r"Low-fidelity",
        ylabel=r"High-fidelity",
        percentage_plot: float = 0.5,
        file_identifier: str = "LF_HF",
    ):
        n_plot = max(1, int(len(samples_X) * percentage_plot))
        print(
            f"Making {xlabel} vs. {ylabel} joint distribution using "
            f"{n_plot}/{len(samples_X)} samples"
        )
        # extract field
        flat_X = samples_X[:n_plot].flatten().numpy()
        flat_Y = samples_Y[:n_plot].flatten().numpy()

        def _plot(x, y, ax):
            assert (
                x.shape == y.shape
            ), f"x ({x.shape}) and y ({y.shape}) must have the same shape"
            r, _ = pearsonr(x, y)

            # Create scatter plot with KDE coloring
            xy = np.vstack([x, y])
            xy = xy + 1e-6 * np.random.randn(*xy.shape)  # add jitter
            z = gaussian_kde(xy)(xy)

            # Sort points by density for better visualization
            idx = z.argsort()
            x_sorted, y_sorted, z_sorted = x[idx], y[idx], z[idx]

            ax.scatter(
                x_sorted,
                y_sorted,
                c=z_sorted,
                s=10,
                cmap="Blues",
                alpha=0.6,
                edgecolors="none",
            )

            # Add diagonal line
            ax.plot([x.min(), x.max()], [x.min(), x.max()], "r--", lw=2)

            # Add correlation text
            ax.text(
                0.05,
                0.95,
                f"$r_{{Pearson}} = {r: .2f}$",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

            return ax

        fig, axs = plt.subplots(1, 1, figsize=(8, 4), dpi=300, layout="compressed")
        _plot(flat_X, flat_Y, axs)
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)

        plt.savefig(file_identifier + "_joint_plot.png", dpi=300)
        plt.close()

    def _comp_average_pearson(
        self, samples_X, samples_Y, xlabel=r"Low-fidelity", ylabel=r"High-fidelity"
    ):
        def _comp_pearson(x: np.ndarray, y: np.ndarray):
            assert x.shape == y.shape, "x and y must have the same shape"
            assert isinstance(x, np.ndarray)
            assert isinstance(y, np.ndarray)
            r, _ = pearsonr(x, y)
            return r

        n_total = len(samples_X)
        r_list = []
        for ii in range(n_total):
            flat_X = samples_X[ii].flatten().numpy()
            flat_Y = samples_Y[ii].flatten().numpy()
            r_list.append(
                _comp_pearson(
                    flat_X,
                    flat_Y,
                )
            )
        assert len(r_list) == n_total, "Number of samples mismatch"

        r_array = np.array(r_list)
        mean_r = np.mean(r_array)
        std_r = np.std(r_array)
        min_r = r_array.min()
        max_r = r_array.max()

        print(
            f"{n_total} sample average {xlabel} vs. {ylabel} "
            f"Pearson correlation coefficient : {mean_r: .4f}  +/- {std_r: .4f} |"
            f" Min: {min_r: .4f} Max: {max_r: .4f}"
        )

    def _compare(self, u_HF, u_LF):
        """compare"""
        if self.plot:
            assert (
                self.n_unique_conditions <= 2000
            ), "For making plots, set unique conditons < 2000"

            # make marginals
            self._make_marginal_plot(
                u_LF=u_LF,
                u_HF=u_HF,
                percentage_plot=0.1,
            )
            # make LF-Residual joint plot
            self._make_joint_plot(
                samples_X=u_LF,
                samples_Y=u_HF - u_LF,
                xlabel=r"Low-fidelity",
                ylabel=r"Residual",
                percentage_plot=0.1,
                file_identifier="Residual",
            )
            # make LF-HF joint plot
            self._make_joint_plot(
                samples_X=u_LF,
                samples_Y=u_HF,
                xlabel=r"Low-fidelity",
                ylabel=r"High-fidelity",
                percentage_plot=0.1,
                file_identifier="LF_HF",
            )
        # compute average pearson for LF-Residual
        self._comp_average_pearson(
            samples_X=u_LF,
            samples_Y=u_HF - u_LF,
            xlabel=r"Low-fidelity",
            ylabel=r"Residual",
        )

        # compute average pearson for LF-HF
        self._comp_average_pearson(
            samples_X=u_LF,
            samples_Y=u_HF,
            xlabel=r"Low-fidelity",
            ylabel=r"High-fidelity",
        )

    def _prep_and_save(self, u_HF, u_LF, a_HF, a_LF):
        # shape to (B, C, *dims)
        u_HF = u_HF.unsqueeze(1)
        u_LF = u_LF.unsqueeze(1)
        a_HF = a_HF.unsqueeze(1)
        a_LF = a_LF.unsqueeze(1)
        # reshape domain to (resolution, 1)
        domain = self.domain.view(-1, 1)
        # prep data dict
        high_data = {
            "field": u_HF,
            "condition": a_HF,
            "field_domain": domain,
            "condition_domain": domain,
            "n_unique_conditions": self.n_unique_conditions,
            "n_fields_per_sample": self.n_fields_per_sample,
        }
        low_data = {
            "field": u_LF,
            "condition": a_LF,
            "field_domain": domain,
            "condition_domain": domain,
            "n_unique_conditions": self.n_unique_conditions,
            "n_fields_per_sample": self.n_fields_per_sample,
        }

        # save
        torch.save(high_data, "high_fidelity.pt")
        torch.save(low_data, "low_fidelity.pt")

        print("saved high fidelity data to high_fidelity.pt")
        print("saved low fidelity data to low_fidelity.pt")

    def _solve_LF(self, a):
        """solve LF"""
        # determinsitic solve
        u_LF_det = self.solver_LF(a)
        # get random field
        random_field = self._get_random_field()
        # apply multiplicative stochasticity
        u_LF = self._apply_multiplicative_random_field(u_LF_det, random_field)
        return u_LF

    def _solve_HF(self, u_LF, a):
        """HF residual = alpha*u_LF + beta*cos(2a) + gamma*z
        - alpha*u_LF:   linear term that sets moderate u_LF correlation
        - beta*cos(2a): second harmonic — new spatial feature absent from LF sin(a)
        - gamma*z:      additive independent GRF noise
        """
        linear_correction = self.alpha * u_LF
        second_harmonic = self.beta * torch.cos(2.0 * a)
        random_field = self._get_random_field()
        u_HF = u_LF + linear_correction + second_harmonic + self.gamma * random_field
        return u_HF

    def simulate(self):
        # get the input conditions: (n_unique_conditions*n_fields_per_sample, N)
        a_LF, a_HF = self._get_input_conditions()
        # solve LF
        u_LF = self._solve_LF(a_LF)
        # solve HF
        u_HF = self._solve_HF(u_LF, a_LF)

        if self.plot:
            self._make_sample_comparison_plot(u_LF=u_LF, u_HF=u_HF)
        # compare
        self._compare(u_HF=u_HF, u_LF=u_LF)
        # correlation between input a and solution
        self._comp_average_pearson(
            samples_X=a_HF,
            samples_Y=u_HF,
            xlabel=r"Input a",
            ylabel=r"High-fidelity solution",
        )
        self._comp_average_pearson(
            samples_X=a_LF,
            samples_Y=u_LF,
            xlabel=r"Input a",
            ylabel=r"Low-fidelity solution",
        )

        # prep and save
        self._prep_and_save(
            u_HF=u_HF,
            u_LF=u_LF,
            a_HF=a_HF,
            a_LF=a_LF,
        )


if __name__ == "__main__":
    # seed
    seed_everything()
    # args
    args = parse_args()
    # Multi-fidelity setup
    mf = MultiFidelity(
        resolution=args.resolution,
        n_unique_conditions=args.n_unique_conditions,
        plot=args.plot,
        rbf_length_scale=args.rbf_length_scale,
        rbf_sigma=args.rbf_sigma,
        rbf_kl_modes=args.rbf_kl_modes,
        n_fields_per_sample=args.n_fields_per_sample,
    )
    mf.simulate()
