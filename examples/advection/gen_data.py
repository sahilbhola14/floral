# examples/advection/gen_data.py
"""
Advection equation
"""
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.stats import pearsonr, gaussian_kde
from scipy.interpolate import RegularGridInterpolator

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
        default=6000,
        help="Number of samples to generate",
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
        default=40,
        help="Number of discretization points for the low-fidelity model",
    )

    parser.add_argument(
        "--Nt",
        type=int,
        default=64,
        help="Number of time discretization points",
    )

    parser.add_argument(
        "--n_modes",
        type=int,
        default=2,
        help="Number of modes",
    )

    args = parser.parse_args()
    print("#" * 50)
    print("Viscous Advection equation")
    print("#" * 50)
    print(f"Number of samples: {args.n_samples}")
    print(f"Number of modes: {args.n_modes}")
    print(f"High-fidelity resolution: {args.resolution_HF}")
    print(f"Low-fidelity resolution: {args.resolution_LF}")
    print("#" * 50)
    return args


def check_blowup(field: np.ndarray, string: str = ""):
    """check if the field has blown up (nan or inf values)"""
    if np.isnan(field).any() or np.isinf(field).any():
        raise ValueError(f"{string} field has blown up!")


class AdvectionSolver:
    """Advection solver.
    Attributes:
    """

    def __init__(
        self,
        beta: float = 0.05,
        Nx: int = 64,
        Nt: int = 64,
        T: float = 1.0,
        Lx: float = 1.0,
        n_modes: int = 2,
        n_samples: int = 100,
        kmax: int = 8,
        use_ic: np.ndarray | None = None,
    ):
        self.beta = beta
        self.Nx = Nx
        self.Nt = Nt
        self.T = T
        self.Lx = Lx
        self.n_modes = n_modes
        self.n_samples = n_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"running solve on: {self.device}")

        self.dt = self.T / self.Nt
        self.x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        self.t = np.arange(0, self.Nt + 1) * self.dt
        self.dx = self.x[1] - self.x[0]
        self.cfl = abs(self.beta) * self.dt / self.dx
        print(f"CFL: {self.cfl: .3e}")
        assert self.cfl < 1.0, "CFL must be less than 1.0 for numerical stability"
        XX, TT = np.meshgrid(self.x, self.t[1:])
        self.domain = np.concatenate((XX.reshape(-1, 1), TT.reshape(-1, 1)), axis=1)
        print(f"dt: {self.dt: .4f} dx: {self.dx: .4f}")

        # initial conditions
        if use_ic is None:
            print("Computing IC")
            self.u0 = self.get_initial_conditions(
                kmax=kmax, n_modes=n_modes, n_samples=self.n_samples
            )
            assert isinstance(self.u0, np.ndarray) and self.u0.shape == (
                self.n_samples,
                self.Nx,
            )
        else:
            print("Using specified IC")
            assert isinstance(use_ic, np.ndarray) and use_ic.shape == (
                self.n_samples,
                self.Nx,
            )
            self.u0 = use_ic
        # check blow up
        check_blowup(self.u0, "Initial condition")

    def get_initial_conditions(self, kmax: int, n_modes: int, n_samples: int):
        """initial condition"""
        np.random.seed(42)

        n_j = np.random.randint(1, kmax + 1, size=(n_samples, n_modes))
        wave_number = 2 * np.pi * n_j / self.Lx

        amp = np.random.uniform(0, 1, size=(n_samples, n_modes))
        phase = np.random.uniform(0, 2 * np.pi, size=(n_samples, n_modes))

        u0 = np.sum(
            amp[:, :, None]
            * np.sin(wave_number[:, :, None] * self.x + phase[:, :, None]),
            axis=1,
        )

        # random abs
        mask = np.random.rand(n_samples) < 0.1
        u0[mask] = np.abs(u0[mask])
        # random sign flip
        mask = np.random.rand(n_samples) < 0.5
        u0[mask] = -u0[mask]
        # windowing
        mask = np.random.rand(n_samples) < 0.1
        n_w = mask.sum()
        if n_w > 0:
            xL = np.random.uniform(0.1, 0.45, size=n_w)
            xR = np.random.uniform(0.55, 0.9, size=n_w)
            trns = 0.1
            window = 0.5 * (
                np.tanh((self.x - xL[:, None]) / trns)
                - np.tanh((self.x - xR[:, None]) / trns)
            )
            u0[mask] *= window

        return u0

    def set_initial_condition(self, use_ic):
        assert isinstance(use_ic, np.ndarray) and use_ic.shape == (
            self.n_samples,
            self.Nx,
        )
        self.u0 = use_ic

    def _rhs(self, u):
        """compute rhs
        For advection term, second order upwind is used and for diffusion terms we use
        central difference.
        """
        assert isinstance(u, torch.Tensor) and u.ndim == 1
        # u_im1 = torch.roll(u, 1)
        # u_ip1 = torch.roll(u, -1)
        # dudx = (u_ip1 - u_im1) / (2.0 * self.dx)
        if self.beta > 0:
            dudx = (u - torch.roll(u, 1)) / self.dx
        else:
            dudx = (torch.roll(u, -1) - u) / self.dx
        return -self.beta * dudx

    def step_rk3(self, u):
        """Third order strong stability preserving Rungu-Kutta (SSPRK3) method
        Reference: https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
        """
        k1 = self._rhs(u)
        k2 = self._rhs(u + self.dt * k1)
        k3 = self._rhs(u + self.dt * (0.25 * k1 + 0.25 * k2))
        un = u + (self.dt / 6.0) * (k1 + k2 + 4.0 * k3)
        return un

    def step_euler(self, u):
        """Forward euler"""
        return u + self.dt * self._rhs(u)

    def comp_exact_solution(self, u0, t):
        assert isinstance(u0, np.ndarray) and u0.ndim == 1
        # Shift by beta*t with periodic wrapping
        shift = (self.x - self.beta * t) % self.Lx
        return np.interp(shift, self.x, u0, period=self.Lx)

    def solve_single(self, u0):
        """Returns (nt + 1, nx) solution"""
        u = torch.tensor(u0, dtype=torch.float32, device=self.device)
        sol_exact = np.zeros((self.Nt + 1, self.Nx))
        sol_exact[0] = u.clone().cpu().numpy()
        sol = torch.zeros(
            (self.Nt + 1, self.Nx), dtype=torch.float32, device=self.device
        )
        sol[0] = u.clone()
        energy = torch.zeros(self.Nt + 1, device=self.device)
        energy[0] = 0.5 * torch.sum(sol[0] * sol[0])
        for ii in range(1, self.Nt + 1):
            u = self.step_euler(u)
            sol[ii] = u.clone()
            energy[ii] = 0.5 * torch.sum(sol[ii] * sol[ii])
            sol_exact[ii] = self.comp_exact_solution(sol_exact[0], ii * self.dt)
        return sol.cpu().numpy(), energy.cpu().numpy(), sol_exact

    def solve(self):
        """all solve"""
        all_uT = []
        all_energy = []
        all_uT_exact = []
        pbar = tqdm(range(self.n_samples), desc="Advection")
        for ii in pbar:
            u0 = self.u0[ii]
            sol, energy, sol_exact = self.solve_single(u0)
            all_uT.append(sol)
            all_energy.append(energy)
            all_uT_exact.append(sol_exact)
        pbar.close()
        all_uT = np.array(all_uT)
        all_uT_exact = np.array(all_uT_exact)
        all_energy = np.array(all_energy)

        # fig, axs = plt.subplots(10, 3, figsize=(6, 20), sharex=True, sharey=True)
        # for ii in range(len(axs)):
        #     vmin = min(all_uT[ii].min(), all_uT_exact[ii].min())
        #     vmax = max(all_uT[ii].max(), all_uT_exact[ii].max())
        #     axs[ii, 0].imshow(all_uT_exact[ii], interpolation="bicubic")
        #     axs[ii, 1].imshow(all_uT[ii], interpolation="bicubic")
        # axs[ii, 2].imshow(np.abs(all_uT[ii], all_uT_exact[ii]),
        #         interpolation="bicubic")
        # plt.savefig("true_compare.png")
        print(f"Error norm: {np.linalg.norm(all_uT - all_uT_exact): .2f}")
        return np.array(all_uT), np.array(all_energy)


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
    ):
        self.n_modes = n_modes
        self.resolution_HF = resolution_HF
        self.resolution_LF = resolution_LF
        self.n_samples = n_samples
        self.Nt = Nt
        self.beta = beta
        assert (
            self.resolution_LF <= self.resolution_HF
        ), "Low-fidelity resolution must be less or equal than high-fidelity"

        # high-fidelity solver
        self.solver_HF = AdvectionSolver(
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
            Nt_LF = int(T / dt_LF)
        else:
            Nt_LF = self.Nt
        self.solver_LF = AdvectionSolver(
            beta=self.beta,
            Nx=self.resolution_LF,
            Nt=Nt_LF,
            n_modes=self.n_modes,
            n_samples=self.n_samples,
            Lx=Lx,
            T=T,
        )
        # update low-fidelity initial conditon
        u0_LF = self._restrict_ic(self.solver_HF.u0)
        self.solver_LF.set_initial_condition(u0_LF)

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
        return u_inter

    def _make_sample_comparison_plot(self, samples_LF, samples_HF, n_samples):
        assert (
            samples_LF.shape == samples_HF.shape
        ), "LF and HF samples must have the same shape"
        assert len(samples_LF) >= n_samples, "Not enough samples to plot"
        # plot testing
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
        vmin = min(samples[:n_samples].min() for samples in [samples_LF, samples_HF])
        vmax = min(samples[:n_samples].max() for samples in [samples_LF, samples_HF])
        for ii in range(n_samples):
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
        plt.savefig("sample_comparison.png", dpi=300)

    def _make_joint_plot(self, samples_LF, samples_HF, random_index: bool = False):
        if random_index:
            sample_index = np.random.randint(0, self.n_samples)
        else:
            sample_index = 0
        print(f"Making joint plot for sample index {sample_index}")
        # extract field
        u_LF = samples_LF[sample_index].flatten()
        u_HF = samples_HF[sample_index].flatten()

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
        _plot(u_LF, u_HF, axs)
        axs.set_xlabel(r"Low-fidelity")
        axs.set_ylabel(r"High-fidelity")

        plt.savefig("joint_plot.png", dpi=300)
        plt.close()

    def _comp_average_pearson(self, samples_LF, samples_HF):
        """compute the average pearson correlation coefficient"""

        def _comp_pearson(x: np.ndarray, y: np.ndarray):
            assert x.shape == y.shape, "x and y must have the same shape"
            r, _ = pearsonr(x, y)
            return r

        r_list = []
        for ii in range(self.n_samples):
            # permeability
            u_HF = samples_HF[ii].flatten()
            u_LF = samples_LF[ii].flatten()
            r_list.append(
                _comp_pearson(
                    u_HF,
                    u_LF,
                )
            )
        assert len(r_list) == self.n_samples, "Number of samples mismatch"

        r_array = np.array(r_list)
        mean_r = np.mean(r_array)
        std_r = np.std(r_array)
        min_r = r_array.min()
        max_r = r_array.max()

        print(
            f"{self.n_samples} sample average Pearson correlation coefficient "
            f": {mean_r: .4f}  +/- {std_r: .4f} | Min: {min_r: .4f} Max: {max_r: .4f}"
        )

    def _prep_and_save(self, samples_LF, samples_HF):
        # high data
        high_data = {
            "field": np.expand_dims(
                samples_HF[:, 1:, :], 1
            ),  # do not save initial condition
            "condition": np.expand_dims(
                np.tile(np.expand_dims(self.solver_HF.u0, 1), (1, self.Nt, 1)), 1
            ),
            "field_domain": self.solver_HF.domain,
            "condition_domain": self.solver_HF.domain,
        }
        # low data
        low_data = {
            "field": np.expand_dims(
                samples_LF[:, 1:, :], 1
            ),  # do not save initial condition
            "condition": np.expand_dims(
                np.tile(np.expand_dims(self.solver_HF.u0, 1), (1, self.Nt, 1)), 1
            ),
            "field_domain": self.solver_HF.domain,
            "condition_domain": self.solver_HF.domain,
        }

        for k, v in high_data.items():
            print(f"Shape of {k} is {v.shape}")
        for k, v in low_data.items():
            print(f"Shape of {k} is {v.shape}")

        # save
        np.savez("high_fidelity.npz", **high_data)
        np.savez("low_fidelity.npz", **low_data)
        print("Data saved to high_fidelity.npz and low_fidelity.npz")
        print("Simulation complete.")

    def simulate(self):
        """simulate multi-fidelity data"""
        # solve high-fidelity
        tic = time.time()
        uT_HF, energy_HF = self.solver_HF.solve()
        elapsed_HF = time.time() - tic
        print(f"Compute time for High-fidelity : {elapsed_HF: .4f} seconds")
        # solve low-fidelty
        tic = time.time()
        uT_LF, energy_LF = self.solver_LF.solve()
        elapsed_LF = time.time() - tic
        print(f"Compute time for Low-fidelity : {elapsed_LF: .4f} seconds")
        print(f"Cost savings with Low-fidelity: {elapsed_HF/elapsed_LF : .4f}")
        # check blowup
        check_blowup(uT_HF, "High-fidelity solution")
        check_blowup(uT_LF, "Low-fidelity solution")
        # interpolate LF to HF grid
        uT_LF_inter = self._interpolate_solution(uT_LF)
        check_blowup(uT_LF_inter, "Interpolated Low-fidelity solution")
        # check shape
        assert uT_HF.shape == (self.n_samples, self.Nt + 1, self.resolution_HF)
        assert uT_LF_inter.shape == (self.n_samples, self.Nt + 1, self.resolution_HF)
        # compare state pltos
        self._make_sample_comparison_plot(
            samples_LF=uT_LF_inter,
            samples_HF=uT_HF,
            n_samples=10,
        )
        # make joint plot
        self._make_joint_plot(
            samples_LF=uT_LF_inter, samples_HF=uT_HF, random_index=False
        )
        # compute average pearson
        self._comp_average_pearson(samples_LF=uT_LF_inter, samples_HF=uT_HF)
        # reshape data for saving
        self._prep_and_save(samples_LF=uT_LF_inter, samples_HF=uT_HF)


if __name__ == "__main__":
    args = parse_args()
    mf = MultiFidelity(
        T=1.0,
        beta=0.05,
        n_modes=args.n_modes,
        resolution_HF=args.resolution_HF,
        resolution_LF=args.resolution_LF,
        Nt=args.Nt,
        n_samples=args.n_samples,
        recalculate_LF_time=True,
    )
    mf.simulate()
