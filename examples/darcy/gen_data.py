# examples/darcy/gen_data.py
"""
Darcy Flow problem with random permeability field.
Modified from: https://github.com/christian-jacobsen/CoCoGen/tree/master
for multi-fidelity training.
Notes:
    1. For default config, average times are
        (a) High-fidelity: 1.49e-2
        (a) Low-fidelity: 6.04e-4,
    resulting in a cost savings of nearly 25 times.
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy.stats import pearsonr, gaussian_kde
from scipy.interpolate import RegularGridInterpolator
from floral.solver import Darcy
from floral.utils import check_tensor_blowup, sample_grf_rbf

plt.style.use("../../scripts/journal.mplstyle")


def parse_args():
    """Parse command line arguments for data generation."""

    parser = argparse.ArgumentParser(
        prog="Darcy",
        description="generate synthetic data for Darcy flow",
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
        "--beta",
        type=float,
        default=0.1,
        help="Scale of additive GRF noise on HF permeability (beta)",
    )
    parser.add_argument(
        "--rbf_length_scale",
        type=float,
        default=0.2,
        help="Length scale for the RBF kernel",
    )
    parser.add_argument(
        "--rbf_sigma",
        type=float,
        default=1.0,
        help="Amplitude for the RBF kernel",
    )
    parser.add_argument(
        "--rbf_kl_modes",
        type=int,
        default=10,
        help="Number of KL modes to retain",
    )

    parser.add_argument(
        "-res_HF",
        "--resolution_HF",
        type=int,
        default=64,
        help="Number of discretization points for the high-fidelity model",
    )

    parser.add_argument(
        "-res_LF",
        "--resolution_LF",
        type=int,
        default=16,
        help="Number of discretization points for the low-fidelity model",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of cores to use during generation",
    )

    parser.add_argument(
        "--ntheta_HF",
        type=int,
        default=32,
        help="Parammeterization dimension of permeability field",
    )

    parser.add_argument(
        "--ntheta_LF",
        type=int,
        default=32,
        help="Parameterization dimension of permeability field",
    )
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    print("==" * 3 + "darcy equation" + "==" * 3)
    print(f"Number of unique conditions: {args.n_unique_conditions}")
    print(f"Number of fields per sample: {args.n_fields_per_sample}")
    print(f"Total samples: {args.n_unique_conditions * args.n_fields_per_sample}")
    print(f"Number of threads: {args.threads}")
    print(f"High-fidelity parameterization dimension: {args.ntheta_HF}")
    print(f"High-fidelity resolution: {args.resolution_HF}")
    print(f"Low-fidelity parameterization dimension: {args.ntheta_LF}")
    print(f"Low-fidelity resolution: {args.resolution_LF}")
    print(f"RBF length scale: {args.rbf_length_scale}")
    print(f"RBF sigma: {args.rbf_sigma}")
    print(f"RBF KL modes: {args.rbf_kl_modes}")
    print("==" * 3 + "==============" + "==" * 3)

    return args


def seed_everything(seed: int = 42):
    """seed everything"""
    random.seed(seed)
    np.random.seed(seed)


class MultiFidelity:
    def __init__(
        self,
        ntheta_HF: int = 128,
        ntheta_LF: int = 32,
        resolution_HF: int = 64,
        resolution_LF: int = 32,
        n_unique_conditions: int = 100,
        n_fields_per_sample: int = 50,
        threads: int = 32,
        plot: bool = False,
        beta: float = 0.05,
        rbf_length_scale: float = 0.2,
        rbf_sigma: float = 1.0,
        rbf_kl_modes: int = 10,
    ):
        self.ntheta_HF = ntheta_HF
        self.ntheta_LF = ntheta_LF
        self.resolution_HF = resolution_HF
        self.resolution_LF = resolution_LF
        self.n_unique_conditions = n_unique_conditions
        self.n_fields_per_sample = n_fields_per_sample
        self.n_samples = n_unique_conditions * n_fields_per_sample
        self.plot = plot
        self.beta = beta
        self.rbf_length_scale = rbf_length_scale
        self.rbf_sigma = rbf_sigma
        self.rbf_kl_modes = rbf_kl_modes

        assert (
            self.ntheta_LF <= self.ntheta_HF
        ), "Low-fidelity parameterization dimension must be less than high-fidelity"
        assert (
            self.resolution_LF <= self.resolution_HF
        ), "Low-fidelity resolution must be less than or equal to high-fidelity"

        # solvers initialised with n_unique_conditions deterministic samples
        self.solver_HF = Darcy(
            resolution=self.resolution_HF,
            ntheta=self.ntheta_HF,
            threads=threads,
            n_samples=self.n_unique_conditions,
        )
        self.solver_LF = Darcy(
            resolution=self.resolution_LF,
            ntheta=self.ntheta_LF,
            threads=threads,
            n_samples=self.n_unique_conditions,
        )

        # build deterministic LF permeability (truncated KL)
        permeability_LF_data = self._create_low_fidelity_permeability_data()
        self.solver_LF._set_permeability_data(permeability_LF_data)

        # tile and add GRF noise to HF; tile (no noise) for LF
        self._get_stochastic_permeability()

    def _restrict_permeability(self, K):
        pbar = tqdm(range(len(K)), desc="Restricting permeability")
        mv = np.zeros((len(K), self.resolution_LF, self.resolution_LF))
        for ii in pbar:
            interpolator = RegularGridInterpolator(
                (self.solver_HF.xv, self.solver_HF.xv),
                K[ii],
                method="linear",
            )
            mv[ii] = interpolator(self.solver_LF.mesh.reshape(-1, 2)).reshape(
                self.resolution_LF, self.resolution_LF
            )

        check_tensor_blowup(mv, "Interpolated permeability")
        return mv

    def _get_random_field(self, seed: int = 42) -> np.ndarray:
        coords = self.solver_HF.mesh.reshape(-1, 2)  # (res_HF^2, 2)
        noise, _ = sample_grf_rbf(
            coords=coords,
            n_samples=self.n_samples,
            length_scale=self.rbf_length_scale,
            sigma=self.rbf_sigma,
            nugget=1e-6,
            rng=np.random.default_rng(seed),
            num_modes=self.rbf_kl_modes,
        )
        return noise.reshape(self.n_samples, self.resolution_HF, self.resolution_HF)

    def _make_permeability_comparison_plot(
        self,
        K_HF_det: np.ndarray,
        K_HF_stoch: np.ndarray,
        K_LF_det: np.ndarray,
        n_plot: int = 4,
    ):
        """3-row grid: det HF | stoch HF | det LF, one column per condition."""
        n_plot = min(n_plot, self.n_unique_conditions)
        rows = ["Det HF", "Stoch HF", "Det LF"]
        fields = [K_HF_det[:n_plot], K_HF_stoch[:n_plot], K_LF_det[:n_plot]]

        fig, axs = plt.subplots(
            3, n_plot, figsize=(3 * n_plot, 8), dpi=150, layout="compressed"
        )
        if n_plot == 1:
            axs = axs[:, None]

        for row, (label, K) in enumerate(zip(rows, fields)):
            for col in range(n_plot):
                im = axs[row, col].imshow(
                    np.log(K[col]), origin="lower", interpolation="bicubic"
                )
                fig.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                if col == 0:
                    axs[row, col].set_ylabel(label)
                if row == 0:
                    axs[row, col].set_title(f"Condition {col}")

        plt.savefig("permeability_comparison.png", dpi=150)
        plt.close()

    def _get_stochastic_permeability(self):
        # deterministic permeabilities: (n_unique_conditions, res, res)
        K_HF_det = self.solver_HF.permeability_data["permeability"]
        K_LF_det = self.solver_LF.permeability_data["permeability"]

        # tile: [c0, c1, ..., c0, c1, ...]
        K_HF_tiled = np.tile(K_HF_det, (self.n_fields_per_sample, 1, 1))
        K_LF_tiled = np.tile(K_LF_det, (self.n_fields_per_sample, 1, 1))
        self.K_HF_det_tiled = K_HF_tiled  # saved for use as condition

        # K_HF
        grf_noise = self._get_random_field()  # (n_samples, res_HF, res_HF)
        K_HF_stoch = K_HF_tiled * (1 + self.beta * grf_noise)
        # Pearson r between deterministic and stochastic HF permeability
        rs = np.array(
            [
                pearsonr(K_HF_tiled[i].ravel(), K_HF_stoch[i].ravel())[0]
                for i in range(self.n_samples)
            ]
        )
        print(
            f"Permeability Pearson (det HF vs stoch HF): mean={rs.mean():.4f} "
            f"+/- {rs.std():.4f} | min={rs.min():.4f} max={rs.max():.4f}"
        )

        if self.plot:
            self._make_permeability_comparison_plot(K_HF_det, K_HF_stoch, K_LF_det)

        # push back into solvers
        self.solver_HF.permeability_data["permeability"] = K_HF_stoch
        self.solver_HF.n_samples = self.n_samples
        self.solver_LF.permeability_data["permeability"] = K_LF_tiled
        self.solver_LF.n_samples = self.n_samples

    def _create_low_fidelity_permeability_data(self):
        permeability_HF_data = self.solver_HF.permeability_data

        ev_LF = permeability_HF_data["eigenvalues"][: self.ntheta_LF]
        U_LF = permeability_HF_data["eigenvectors"][:, : self.ntheta_LF]
        theta_LF = permeability_HF_data["generative_params"][: self.ntheta_LF, :]

        # construct low-fidelity permeability fields (B, resolution_HF, resolution_HF)
        K_LF = self.solver_LF._construct_permeability_from_UL(ev_LF, U_LF, theta_LF)
        # restrict to the low-fidelity resolution (B, resolution_LF, resolution_LF)
        K_LF_int = self._restrict_permeability(K_LF)

        permeability_LF_data = {
            "permeability": K_LF_int,
            "generative_params": theta_LF,
            "eigenvalues": ev_LF,
            "eigenvectors": U_LF,
        }

        return permeability_LF_data

    def simulate_model(self, model="HF"):
        if model == "HF":
            data_dict = self.solver_HF.solve()
        elif model == "LF":
            data_dict = self.solver_LF.solve()
        else:
            raise ValueError(f"Invalid model selected: {model}")
        return data_dict

    def _interpolate_solution(self, data_dict):
        data_dict_mv = {}
        for k, ov in data_dict.items():
            mv = np.zeros((self.n_samples, self.resolution_HF, self.resolution_HF))

            pbar = tqdm(range(len(ov)), desc=f"Interpolating {k}")
            for ii in pbar:
                interpolator = RegularGridInterpolator(
                    (self.solver_LF.xv, self.solver_LF.xv),
                    ov[ii],
                    method="linear",
                )
                mv[ii] = interpolator(self.solver_HF.mesh.reshape(-1, 2)).reshape(
                    self.resolution_HF, self.resolution_HF
                )

            data_dict_mv[k] = mv
            assert mv.shape == (self.n_samples, self.resolution_HF, self.resolution_HF)

        return data_dict_mv

    def _make_marginal_plot(self, dict_HF, dict_LF, percentage_plot: int = 0.5):
        """marginal density plot"""
        n_plot = max(1, int(self.n_samples * percentage_plot))
        print(f"Making marginal distribution using {n_plot}/{self.n_samples} samples")
        fig, axs = plt.subplots(2, 2, figsize=(6, 6), layout="compressed")
        # permeability
        flat_K_LF = dict_LF["permeability"][:n_plot].flatten()
        flat_K_HF = dict_HF["permeability"][:n_plot].flatten()
        flat_K_res = flat_K_HF - flat_K_LF
        # pressure
        flat_P_LF = dict_LF["pressure"][:n_plot].flatten()
        flat_P_HF = dict_HF["pressure"][:n_plot].flatten()
        flat_P_res = flat_P_HF - flat_P_LF

        axs[0, 0].hist(flat_K_LF, density=True, bins=100, label=r"Low-fidelity")
        axs[0, 0].hist(flat_K_HF, density=True, bins=100, label=r"High-fidelity")
        axs[0, 0].set_title(r"Permeability, $K$")

        axs[1, 0].hist(flat_P_LF, density=True, bins=100, label=r"Low-fidelity")
        axs[1, 0].hist(flat_P_HF, density=True, bins=100, label=r"High-fidelity")
        axs[1, 0].set_title(r"Pressure, $P$")

        axs[0, 1].hist(flat_K_res, density=True, bins=100, label=r"Permeability")
        axs[0, 1].set_title(r"Residual")
        axs[1, 1].hist(flat_P_res, density=True, bins=100, label=r"Pressure")
        axs[1, 1].set_title(r"Residual")

        for ax in axs.flatten():
            ax.legend()

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
        assert (
            n_plot <= 20
        ), "For making joint distribution plots tractably, reduce the percentation plot"
        print(
            f"Making {xlabel} vs. {ylabel} joint distribution using "
            f"{n_plot}/{len(samples_X)} samples"
        )
        # extract field
        flat_X = samples_X[:n_plot].flatten()
        flat_Y = samples_Y[:n_plot].flatten()

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
        raise_error = False
        for ii in range(self.n_samples):
            flat_X = samples_X[ii].flatten()
            flat_Y = samples_Y[ii].flatten()
            if flat_X.std() < 1e-6 or flat_Y.std() < 1e-6:
                raise_error = True
                fig, axs = plt.subplots(1, 2, figsize=(4, 2))
                axs[0].imshow(flat_X.reshape(self.resolution_HF, self.resolution_HF))
                axs[0].set_title(xlabel)
                axs[1].imshow(flat_Y.reshape(self.resolution_HF, self.resolution_HF))
                axs[1].set_title(ylabel)
                plt.savefig(f"no_variation_{xlabel}_vs_{ylabel}_{ii}.png")
            r_list.append(
                _comp_pearson(
                    flat_X,
                    flat_Y,
                )
            )

        if raise_error:
            raise RuntimeError("No variation found in the solution field.")
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

    def _make_sample_comparison_plot(
        self, samples_LF, samples_HF, file_identifier, n_conditions: int = 4
    ):
        """Plot LF (deterministic) | HF mean | HF std, one row per unique condition."""
        nc = self.n_unique_conditions
        nf = self.n_fields_per_sample
        n_conditions = min(n_conditions, nc)

        cols = [
            "Low-fidelity",
            "High-fidelity mean",
            "High-fidelity std",
            "Residual mean",
        ]
        fig, axs = plt.subplots(
            n_conditions,
            4,
            figsize=(9, 2.5 * n_conditions),
            dpi=300,
            layout="compressed",
        )
        if n_conditions == 1:
            axs = axs[None, :]

        for row, cond_idx in enumerate(range(n_conditions)):
            hf_block = samples_HF[cond_idx::nc][:nf]  # (nf, H, W)
            lf_field = samples_LF[cond_idx]  # (H, W) — deterministic
            hf_mean = hf_block.mean(axis=0)
            hf_std = hf_block.std(axis=0)
            residual_mean = hf_mean - lf_field
            res_abs = np.abs(residual_mean).max()

            vmin = min(lf_field.min(), hf_mean.min())
            vmax = max(lf_field.max(), hf_mean.max())

            panels = [
                (lf_field, vmin, vmax, None),
                (hf_mean, vmin, vmax, None),
                (hf_std, 0, hf_std.max(), None),
                (residual_mean, -res_abs, res_abs, "RdBu_r"),
            ]
            for col, (field, lo, hi, cmap) in enumerate(panels):
                im = axs[row, col].imshow(
                    field,
                    origin="lower",
                    vmin=lo,
                    vmax=hi,
                    interpolation="bicubic",
                    cmap=cmap,
                )
                fig.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                axs[row, col].set_xlabel(r"$x_1$")
                axs[row, col].set_ylabel(r"$x_2$")
                if row == 0:
                    axs[row, col].set_title(cols[col])

        fig.align_labels()
        plt.savefig(file_identifier + "_data_snapshot.png", dpi=300)
        plt.close()

    def _compare(self, dict_HF: dict, dict_LF: dict):
        if self.plot:
            # make sample comparison
            self._make_sample_comparison_plot(
                samples_LF=np.log(dict_LF["permeability"]),
                samples_HF=np.log(dict_HF["permeability"]),
                file_identifier="Log_Permeability",
            )
            self._make_sample_comparison_plot(
                samples_LF=dict_LF["pressure"],
                samples_HF=dict_HF["pressure"],
                file_identifier="Pressure",
            )
            # make marginals
            self._make_marginal_plot(
                dict_HF=dict_HF, dict_LF=dict_LF, percentage_plot=0.1
            )
            # make joint plot(s)
            self._make_joint_plot(
                samples_X=np.log(dict_LF["permeability"]),
                samples_Y=np.log(dict_HF["permeability"]),
                xlabel=r"Low-fidelity",
                ylabel=r"High-fidelity",
                percentage_plot=0.001,
                file_identifier="LF_HF_Log_Permeability",
            )
            self._make_joint_plot(
                samples_X=dict_LF["pressure"],
                samples_Y=dict_HF["pressure"],
                xlabel=r"Low-fidelity",
                ylabel=r"High-fidelity",
                percentage_plot=0.001,
                file_identifier="LF_HF_Pressure",
            )

            self._make_joint_plot(
                samples_X=np.log(dict_LF["permeability"]),
                samples_Y=np.log(dict_HF["permeability"])
                - np.log(dict_LF["permeability"]),
                xlabel=r"Low-fidelity",
                ylabel=r"Residual",
                percentage_plot=0.001,
                file_identifier="Residual_Permeability",
            )

            self._make_joint_plot(
                samples_X=dict_LF["pressure"],
                samples_Y=dict_HF["pressure"] - dict_LF["pressure"],
                xlabel=r"Low-fidelity",
                ylabel=r"Residual",
                percentage_plot=0.001,
                file_identifier="Residual_Pressure",
            )

        # compare averate pearson for LF-Residual
        self._comp_average_pearson(
            samples_X=dict_LF["pressure"],
            samples_Y=dict_HF["pressure"] - dict_LF["pressure"],
            xlabel=r"Low-fidelity",
            ylabel=r"Residual",
        )

        self._comp_average_pearson(
            samples_X=dict_LF["pressure"],
            samples_Y=dict_HF["pressure"],
            xlabel=r"Low-fidelity",
            ylabel=r"High-fidelity",
        )

    def _prep_and_save(self, dict_LF: dict, dict_HF: dict):
        # convert to tensors and reshape
        for k, v in dict_LF.items():
            dict_LF[k] = torch.Tensor(dict_LF[k]).unsqueeze(1)

        for k, v in dict_HF.items():
            dict_HF[k] = torch.Tensor(dict_HF[k]).unsqueeze(1)

        domain = torch.Tensor(self.solver_HF.mesh)
        condition = torch.log(torch.Tensor(self.K_HF_det_tiled).unsqueeze(1))
        high_data = {
            "field": dict_HF["pressure"],
            "condition": condition,
            "field_domain": domain,
            "condition_domain": domain,
            "n_unique_conditions": self.n_unique_conditions,
            "n_fields_per_sample": self.n_fields_per_sample,
        }

        low_data = {
            "field": dict_LF["pressure"],
            "condition": condition,
            "field_domain": domain,
            "condition_domain": domain,
            "n_unique_conditions": self.n_unique_conditions,
            "n_fields_per_sample": self.n_fields_per_sample,
        }

        torch.save(high_data, "high_fidelity.pt")
        torch.save(low_data, "low_fidelity.pt")

        print("saved high fidelity data to high_fidelity.pt")
        print("saved low fidelity data to low_fidelity.pt")

    def simulate(self):
        """simulate multi-fidelity data"""
        # solve high-fidelity
        data_dict_HF = self.simulate_model(model="HF")
        # solve low-fidelity
        data_dict_LF = self.simulate_model(model="LF")
        # interpolate
        data_dict_LF_inter = self._interpolate_solution(data_dict_LF)
        # compare
        self._compare(
            dict_HF=data_dict_HF,
            dict_LF=data_dict_LF_inter,
        )
        # prep and save
        self._prep_and_save(
            dict_HF=data_dict_HF,
            dict_LF=data_dict_LF_inter,
        )


if __name__ == "__main__":
    # seed
    seed_everything()
    # args
    args = parse_args()
    mf = MultiFidelity(
        ntheta_HF=args.ntheta_HF,
        ntheta_LF=args.ntheta_LF,
        resolution_HF=args.resolution_HF,
        resolution_LF=args.resolution_LF,
        n_unique_conditions=args.n_unique_conditions,
        n_fields_per_sample=args.n_fields_per_sample,
        threads=args.threads,
        plot=args.plot,
        beta=args.beta,
        rbf_length_scale=args.rbf_length_scale,
        rbf_sigma=args.rbf_sigma,
        rbf_kl_modes=args.rbf_kl_modes,
    )
    mf.simulate()
