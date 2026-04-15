# /examples/oneDCorr/gen_data.py
"""
a: known input
random_field (unobserved at inference): latent
w: observation
goal: a -> w
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


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(
        prog="onedcorr",
        description="generate synthetic data for onedcorr",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=5000,
        help="Number of samples to generate",
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
        default=0.1,
        help="Lenght scale for the RBF kernel",
    )
    parser.add_argument(
        "--rbf_sigma",
        type=float,
        default=1,
        help="Amplitude for the RBF kernel",
    )

    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    print("==" * 3 + "onedcorr" + "==" * 3)
    print(f"Number of samples: {args.n_samples}")
    print(f"Resolution: {args.resolution}")
    print("==" * 3 + "========" + "==" * 3)

    return args


def seed_everything(seed: int = 42):
    """seed everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiFidelity:
    def __init__(
        self, resolution, n_samples, rbf_length_scale, rbf_sigma, plot: bool = False
    ):
        self.resolution = resolution
        self.n_samples = n_samples
        self.rbf_length_scale = rbf_length_scale
        self.rbf_sigma = rbf_sigma
        self.plot = plot

        self.solver_HF = OneDCorr(
            fidelity="High", resolution=self.resolution, n_samples=n_samples
        )
        self.solver_LF = OneDCorr(
            fidelity="Low", resolution=self.resolution, n_samples=n_samples
        )
        self.domain = self.solver_HF.domain

    def _get_input_conditions(self):
        """same HF and LF conditions"""
        a_HF = self.solver_HF._sample_input_function()
        return a_HF, a_HF

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

    def _get_random_field(self):
        # extract the domain for the HF model
        coords = np.array(self.domain).T  # (N,1)
        random_field = sample_grf_rbf(
            coords=coords,
            n_samples=self.n_samples,
            length_scale=self.rbf_length_scale,
            sigma=self.rbf_sigma,
            nugget=1e-6,
        )
        return torch.tensor(random_field, dtype=torch.float32)

    def _make_sample_comparison_plot(self, u_LF, u_HF, n_samples: int = 5):
        """make sample comparison plot"""
        assert u_LF.shape == u_HF.shape, "LF and HF samples must have the same shape"
        assert len(u_LF) >= n_samples, "Not enough samples to plot"

        fig, axs = plt.subplots(
            n_samples, 2, figsize=(6, n_samples * 2), sharex=True, layout="compressed"
        )
        for ii, ax in enumerate(axs):
            ax[0].plot(self.domain.ravel(), u_LF[ii], color="grey", alpha=0.6)
            ax[0].plot(self.domain.ravel(), u_HF[ii], color="k")
            ax[1].plot(self.domain.ravel(), torch.abs(u_HF[ii] - u_LF[ii]), color="k")
            ax[0].set_xlabel(r"$x$")
            ax[0].set_ylabel(r"$w(x)$")
            ax[1].set_xlabel(r"$x$")
            ax[1].set_ylabel(r"$|w_{HF}(x) - w_{LF}(x)|$")
            # for a in ax:
            #     a.label_outer()
        plt.savefig("data_snapshot.png")

    def _make_marginal_plot(self, u_LF, u_HF, percentage_plot: int = 0.5):
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
        percentage_plot: int = 0.5,
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

        r_list = []
        for ii in range(self.n_samples):
            flat_X = samples_X[ii].flatten().numpy()
            flat_Y = samples_Y[ii].flatten().numpy()
            r_list.append(
                _comp_pearson(
                    flat_X,
                    flat_Y,
                )
            )
        assert len(r_list) == self.n_samples, "Number of samples mismatch"

        r_array = np.array(r_list)
        mean_r = np.mean(r_array)
        std_r = np.std(r_array)
        min_r = r_array.min()
        max_r = r_array.max()

        print(
            f"{self.n_samples} sample average {xlabel} vs. {ylabel} "
            f"Pearson correlation coefficient : {mean_r: .4f}  +/- {std_r: .4f} |"
            f" Min: {min_r: .4f} Max: {max_r: .4f}"
        )

    def _compare(self, u_HF, u_LF):
        """compare"""
        if self.plot:
            assert (
                self.n_samples <= 2000
            ), "For making plots, reduce the number of samples to less than 2000"

            # make sample comparion
            self._make_sample_comparison_plot(
                u_LF=u_LF,
                u_HF=u_HF,
            )
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
        }
        low_data = {
            "field": u_LF,
            "condition": a_LF,
            "field_domain": domain,
            "condition_domain": domain,
        }

        # save
        torch.save(high_data, "high_fidelity.pt")
        torch.save(low_data, "low_fidelity.pt")

        print("saved high fidelity data to high_fidelity.pt")
        print("saved low fidelity data to low_fidelity.pt")

    def simulate(self):
        # get the input conditions
        a_HF, a_LF = self._get_input_conditions()
        # sample the random field (latent)
        random_field = self._get_random_field()
        if self.plot:
            self._plot_random_field(random_field)
        # solve
        u_HF = self.solver_HF(a_HF + random_field)
        u_LF = self.solver_LF(a_LF + random_field)
        # compare
        self._compare(u_HF=u_HF, u_LF=u_LF)
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
        n_samples=args.n_samples,
        plot=args.plot,
        rbf_length_scale=args.rbf_length_scale,
        rbf_sigma=args.rbf_sigma,
    )
    mf.simulate()
