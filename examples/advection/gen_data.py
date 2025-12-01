# examples/advection/gen_data.py
"""
Advection equation
"""
import random
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import torch.fft as fft
from scipy.stats import pearsonr, gaussian_kde
from scipy.interpolate import RegularGridInterpolator
from floral.solver import Advection
from floral.utils import check_tensor_blowup

plt.style.use("../../scripts/journal.mplstyle")


def parse_args():
    """Parse command line arguments for data generation."""

    parser = argparse.ArgumentParser(
        prog="Advection",
        description="generate synthetic data for Advection flow",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=10000,
        help="Number of samples to generate",
    )

    parser.add_argument(
        "-res_HF",
        "--resolution_HF",
        type=int,
        default=128,
        help="Number of discretization points for the high-fidelity model",
    )

    parser.add_argument(
        "-res_LF",
        "--resolution_LF",
        type=int,
        default=128,
        help="Number of discretization points for the low-fidelity model",
    )

    parser.add_argument(
        "--Nt",
        type=int,
        default=128,
        help="Number of time discretization points",
    )

    parser.add_argument(
        "--n_modes",
        type=int,
        default=2,
        help="Number of modes",
    )

    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    print("==" * 3 + "advection" + "==" * 3)
    print(f"Number of samples: {args.n_samples}")
    print(f"Number of modes: {args.n_modes}")
    print(f"High-fidelity Resolution: {args.resolution_HF}")
    print(f"Low-fidelity Resolution: {args.resolution_LF}")
    print("==" * 3 + "=========" + "==" * 3)

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
        self,
        n_modes: int = 2,
        resolution_HF: int = 64,
        resolution_LF: int = 32,
        n_samples: int = 1,
        Nt: int = 64,
        threads: int = 32,
        beta: float = 0.05,
        Lx: float = 1.0,
        T: float = 0.2,
        recalculate_LF_time: bool = False,
        plot: bool = False,
    ):
        self.n_modes = n_modes
        self.resolution_HF = resolution_HF
        self.resolution_LF = resolution_LF
        self.n_samples = n_samples
        self.Nt = Nt
        self.beta = beta
        self.plot = plot
        assert (
            self.resolution_LF <= self.resolution_HF
        ), "Low-fidelity resolution must be less or equal than high-fidelity"
        # high-fidelity solver
        self.solver_HF = Advection(
            beta=self.beta,
            Nx=self.resolution_HF,
            Nt=self.Nt,
            n_modes=self.n_modes,
            n_samples=self.n_samples,
            Lx=Lx,
            T=T,
        )
        ratio = self.solver_HF.dt / self.solver_HF.dx
        # low-fidelity solver
        if recalculate_LF_time:
            dt_LF = ratio * (Lx / self.resolution_LF)
            self.Nt_LF = int(T / dt_LF)
        else:
            self.Nt_LF = self.Nt
        self.solver_LF = Advection(
            beta=self.beta,
            Nx=self.resolution_LF,
            Nt=self.Nt_LF,
            n_modes=self.n_modes,
            n_samples=self.n_samples,
            Lx=Lx,
            T=T,
        )

        # create low fidelity inital condition
        u0_LF = self._create_low_fidelity_ic()
        # update low fidelity initial condition
        self.solver_LF.set_initial_condition(u0_LF)

    def _create_low_fidelity_ic(self, keep_frac=0.4, amp_scale=0.8, gamma=4.0):
        # HF initial condition
        u0_HF = self.solver_HF.u0  # (B, res_HF)
        # spectral filter
        u0_LF = self._filter_ic(
            u0_HF, keep_frac=keep_frac, amp_scale=amp_scale, gamma=gamma
        )
        # restrict to the Low-fidelity domain
        u0_LF = self._restrict_ic(u0_LF)
        return u0_LF

    def _filter_ic(
        self,
        u0_HF: np.ndarray,
        keep_frac: float = 0.4,
        amp_scale: float = 0.8,
        gamma: float = 4.0,
        phase_bias: float = 0.0,
        min_kept_modes: int = 2,
    ) -> torch.Tensor:
        """
        u0_HF: (B, N) high-fidelity initial condition
        keep_frac: fraction of lowest frequencies to keep
        amp_scale: global amplitude scaling for LF
        gamma: strength of exponential damping for kept modes
        phase_bias: magnitude of random phase distortion
        min_kept_modes: minimum number of modes to keep
        """
        u0_HF = torch.Tensor(u0_HF)
        B, N = u0_HF.shape
        # FFT
        u_hat = fft.rfft(u0_HF, dim=-1)  # (B, N//2+1)
        n_modes = u_hat.shape[-1]

        # keep low modes
        k_keep = max(min_kept_modes, int(keep_frac * n_modes))
        mask = torch.zeros_like(u_hat)
        mask[..., :k_keep] = 1.0

        # mode dependent amplitude bias
        k = torch.arange(n_modes, device=u_hat.device)[None, :]
        alpha = torch.exp(-gamma * (k / n_modes))  # smooth bias

        # phase bias (structural misalignment)
        if phase_bias > 0:
            phase_noise = phase_bias * torch.randn_like(u_hat)  # real-valued
            phase_factor = torch.exp(1j * phase_noise)  # abs value =1
        else:
            phase_factor = 1.0

        # LF spectrum
        u_hat_LF = amp_scale * (u_hat * mask * alpha) * phase_factor
        # u_hat_LF = amp_scale * (u_hat * mask)

        # inverse fft
        u0_LF = fft.irfft(u_hat_LF, n=N, dim=-1)

        # frequencies (for plot)
        N = u_hat.shape[-1] * 2 - 2
        freqs = torch.fft.rfftfreq(N)

        assert u0_LF.std(1).min() > 0, "No variation in IC"

        if self.plot:
            assert self.n_samples >= 4, "4 initial condtions are plotted"
            fig, axs = plt.subplots(2, 4, layout="compressed", figsize=(13, 5))
            # start_plot = np.random.randint(0, high=len(u_hat))
            start_plot = 69
            for ii in range(4):
                # spectral
                u_hat_HF_mag = torch.abs(u_hat)[start_plot + ii]
                u_hat_LF_mag = torch.abs(u_hat_LF)[start_plot + ii]
                # physical
                u_HF = u0_HF[start_plot + ii]
                u_LF = u0_LF[start_plot + ii]

                axs[0, ii].plot(freqs, u_hat_HF_mag, label="High-fidelity", color="k")
                axs[0, ii].plot(freqs, u_hat_LF_mag, label="Low-fidelity", color="red")
                axs[0, ii].set_xlabel("Frequency")
                axs[0, ii].set_ylabel(r"Amplitude $|\hat{u}[k]|$")

                axs[1, ii].plot(
                    self.solver_HF.x.ravel(), u_HF, label="High-fidelity", color="k"
                )
                axs[1, ii].plot(
                    self.solver_HF.x.ravel(), u_LF, label="Low-fidelity", color="red"
                )
                axs[1, ii].set_xlabel("$x$")
                axs[1, ii].set_ylabel(r"$u(x, 0)$")

                if ii == 0:
                    axs[0, ii].legend(loc="upper right")
            fig.align_labels()
            plt.savefig("fourier_initial_condition_comparison.png")
        return u0_LF.numpy()

    def _restrict_ic(self, u0_HF):
        """restrict the ic of High-fidelity to low-fidelity domain"""
        u0_LF = []
        for u0 in u0_HF:
            interpolator = RegularGridInterpolator(
                (self.solver_HF.x,),
                u0.reshape(-1),
                method="linear",
            )
            u0_LF.append(interpolator(self.solver_LF.x))
        u0_LF = np.stack(u0_LF)
        assert u0_LF.shape == (self.n_samples, self.resolution_LF)
        return u0_LF

    def _interpolate_solution(self, uT):
        u_inter = []
        t_HF = self.solver_HF.t
        x_HF = self.solver_HF.x
        # make meshgrid of HF grid
        Tq, Xq = np.meshgrid(t_HF, x_HF, indexing="ij")  # shape (Nt_HF+1, nx_HF)
        query_points = np.column_stack(
            [Tq.ravel(), Xq.ravel()]
        )  # shape (Nt_HF+1 * nx_HF, 2)

        for u in uT:
            interpolator = RegularGridInterpolator(
                (self.solver_LF.t, self.solver_LF.x),
                u,
                method="linear",
                bounds_error=False,
                fill_value=None,  # extrapolation
            )
            u_interp_flat = interpolator(query_points)  # shape flattened
            u_interp = u_interp_flat.reshape(Tq.shape)  # reshape back
            u_inter.append(u_interp)
        u_inter = np.stack(u_inter)
        assert u_inter.shape == (self.n_samples, self.Nt + 1, self.resolution_HF)

        check_tensor_blowup(u_inter, "Interpolated solution")

        return u_inter

    def _make_sample_comparison_plot(self, samples_LF, samples_HF, n_samples):
        assert (
            samples_LF.shape == samples_HF.shape
        ), "LF and HF samples must have the same shape"
        assert len(samples_LF) >= n_samples, "Not enough samples to plot"

        fig, axs = plt.subplots(
            n_samples,
            3,
            figsize=(6, 2 * n_samples),
            dpi=300,
            layout="compressed",
            sharex=True,
            sharey=True,
        )
        axs_lf = axs[:, 0]
        axs_hf = axs[:, 1]
        axs_diff = axs[:, 2]
        # vmin = min(samples[:n_samples].min() for samples in [samples_LF, samples_HF])
        # vmax = min(samples[:n_samples].max() for samples in [samples_LF, samples_HF])
        for ii in range(n_samples):

            vmin = min(samples_LF[ii].min(), samples_HF[ii].min())
            vmax = max(samples_LF[ii].max(), samples_HF[ii].max())

            im_field = axs_lf[ii].imshow(
                samples_LF[ii],
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                interpolation="bicubic",
            )
            axs_hf[ii].imshow(
                samples_HF[ii],
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                interpolation="bicubic",
            )
            im_diff = axs_diff[ii].imshow(
                np.abs(samples_HF[ii] - samples_LF[ii]),
                origin="lower",
                interpolation="bicubic",
            )
            fig.colorbar(
                im_field, ax=axs_hf[ii], orientation="vertical", fraction=0.046, pad=0.1
            )
            fig.colorbar(
                im_diff,
                ax=axs_diff[ii],
                orientation="vertical",
                fraction=0.046,
                pad=0.1,
            )
            if ii == 0:
                axs_lf[ii].set_title(r"Low-fidelity")
                axs_hf[ii].set_title(r"High-fidelity")
                axs_diff[ii].set_title(r"Absolute error")
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$t$")
            ax.label_outer()
        plt.savefig("data_snapshot.png", dpi=300)

    def _make_marginal_plot(self, u_LF, u_HF, percentage_plot: int = 0.5):
        n_plot = max(1, int(len(u_LF) * percentage_plot))
        print(f"Making marginal distribution using {n_plot}/{len(u_LF)} samples")
        # extract field
        flat_LF = u_LF[:n_plot].flatten()
        flat_HF = u_HF[:n_plot].flatten()
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
        assert (
            n_plot <= 5
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

    def _make_initial_condition_plot(self):
        """make initial condition plot"""
        fig, axs = plt.subplots(
            5, 2, sharex=True, sharey=True, layout="compressed", figsize=(6, 10)
        )
        for ii, ax in enumerate(axs.flatten()):
            ax.plot(
                self.solver_HF.x,
                self.solver_HF.u0[ii],
                label="High-fidelity",
                color="k",
            )
            ax.plot(
                self.solver_LF.x,
                self.solver_LF.u0[ii],
                label="Low-fidelity",
                color="grey",
                alpha=0.6,
                linestyle="--",
            )
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$u(x, 0)$")
            ax.label_outer()
            if ii == 0:
                ax.legend(loc="lower right")
        plt.savefig("initial_condition.png")

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
            flat_X = samples_X[ii].flatten()
            flat_Y = samples_Y[ii].flatten()
            if flat_X.std() < 1e-6 or flat_Y.std() < 1e-6:
                raise RuntimeError("No variation found in the solution field.")
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

    def _prep_and_save(self, u_LF, u_HF):
        # convert to tensor and reshape to (B, C, *dims)
        u_HF = torch.Tensor(u_HF).unsqueeze(1)
        u_LF = torch.Tensor(u_LF).unsqueeze(1)
        # remove the initial condition
        u_HF = u_HF[:, :, 1:, :]
        u_LF = u_LF[:, :, 1:, :]
        # prepare the condition (tiled initial conditon)
        condition = (
            torch.Tensor(self.solver_HF.u0)
            .unsqueeze(1)
            .expand(-1, self.Nt, -1)
            .unsqueeze(1)
        )
        domain = torch.Tensor(self.solver_HF.domain)

        # prep data dict
        high_data = {
            "field": u_HF,
            "condition": condition,
            "field_domain": domain,
            "condition_domain": domain,
        }
        low_data = {
            "field": u_LF,
            "condition": condition,
            "field_domain": domain,
            "condition_domain": domain,
        }

        # save
        torch.save(high_data, "high_fidelity.pt")
        torch.save(low_data, "low_fidelity.pt")

        print("saved high fidelity data to high_fidelity.pt")
        print("saved low fidelity data to low_fidelity.pt")

    def simulate_model(self, model="HF"):
        tic = time.time()
        if model == "HF":
            uT, energy = self.solver_HF.solve()
            assert uT.shape == (self.n_samples, self.Nt + 1, self.resolution_HF)
            assert energy.shape == (self.n_samples, self.Nt + 1)
        elif model == "LF":
            uT, energy = self.solver_LF.solve()
            assert uT.shape == (self.n_samples, self.Nt_LF + 1, self.resolution_LF)
            assert energy.shape == (self.n_samples, self.Nt_LF + 1)
        else:
            raise ValueError(f"Invalid model: {model}")
        elapsed_HF = time.time() - tic
        print(f"Compute time for {model} : {elapsed_HF: .4f} seconds")
        check_tensor_blowup(uT, "state")
        check_tensor_blowup(energy, "energy")
        return uT, energy

    def _compare(self, uT_HF: np.ndarray, uT_LF: np.ndarray):
        if self.plot:
            # initial condition plot
            self._make_initial_condition_plot()
            # make sample comparison
            self._make_sample_comparison_plot(
                samples_LF=uT_LF,
                samples_HF=uT_HF,
                n_samples=5,
            )
            # make marginals
            self._make_marginal_plot(u_LF=uT_LF, u_HF=uT_HF, percentage_plot=0.1)
            # make LF-Residual joint plot
            self._make_joint_plot(
                samples_X=uT_LF,
                samples_Y=uT_HF - uT_LF,
                xlabel=r"Low-fidelity",
                ylabel=r"Residual",
                percentage_plot=0.0005,
                file_identifier="Residual",
            )
            # make LF-HF joint plot
            self._make_joint_plot(
                samples_X=uT_LF,
                samples_Y=uT_HF,
                xlabel=r"Low-fidelity",
                ylabel=r"High-fidelity",
                percentage_plot=0.0005,
                file_identifier="LF_HF",
            )
        # compute average pearson for LF-Residual
        self._comp_average_pearson(
            samples_X=uT_LF,
            samples_Y=uT_HF - uT_LF,
            xlabel=r"Low-fidelity",
            ylabel=r"Residual",
        )

        # compute average pearson for LF-HF
        self._comp_average_pearson(
            samples_X=uT_LF,
            samples_Y=uT_HF,
            xlabel=r"Low-fidelity",
            ylabel=r"High-fidelity",
        )

    def simulate(self):
        """simulate multi-fidelity data"""
        # solve high-fidelity
        uT_HF, energy_HF = self.simulate_model("HF")
        # solve low-fidelty
        uT_LF, energy_LF = self.simulate_model("LF")
        # interpolate LF to HF grid
        uT_LF_inter = self._interpolate_solution(uT_LF)
        # compare
        self._compare(uT_HF=uT_HF, uT_LF=uT_LF_inter)
        # prep and save
        self._prep_and_save(u_LF=uT_LF_inter, u_HF=uT_HF)


if __name__ == "__main__":
    # seed
    seed_everything()
    # args
    args = parse_args()
    # Multi-fidelity setup
    mf = MultiFidelity(
        T=1.0,
        beta=0.05,
        n_modes=args.n_modes,
        resolution_HF=args.resolution_HF,
        resolution_LF=args.resolution_LF,
        Nt=args.Nt,
        n_samples=args.n_samples,
        recalculate_LF_time=True,
        plot=args.plot,
    )
    mf.simulate()
