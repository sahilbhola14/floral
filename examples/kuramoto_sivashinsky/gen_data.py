# examples/kuramoto_sivashinsky/gen_data.py
"""
Python script to generate multi-fidelity KS equation data.
Author: Sahil Bhola, University of Michigan, 2025
"""
import numpy as np
import jax
import jax_cfd.base as cfd
import matplotlib.pyplot as plt
import argparse
from jax_cfd.spectral import equations as spectral_equations
from jax import numpy as jnp
from jax_cfd.spectral import time_stepping
from jax import image
from scipy.stats import pearsonr, gaussian_kde

plt.style.use("../../scripts/journal.mplstyle")


def parse_args():
    """Parse command line arguments for data generation."""

    parser = argparse.ArgumentParser(
        prog="KS",
        description="generate synthetic data for KuramotoSivashinsky",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=10000,
        help="Number of samples to generate",
    )

    parser.add_argument(
        "-resHF",
        "--resolution_HF",
        type=int,
        default=256,
        help="Number of discretization points for high-fidelity",
    )

    parser.add_argument(
        "-resLF",
        "--resolution_LF",
        type=int,
        default=48,
        help="Number of discretization points for low-fidelity",
    )
    parser.add_argument(
        "-tsteps",
        "--time_steps",
        type=int,
        default=256,
        help="Number of time steps",
    )
    args = parser.parse_args()
    print("#" * 50)
    print("Kuramoto Sivashinsky")
    print("#" * 50)
    print(f"Number of samples: {args.n_samples}")
    print(f"Resolution for High-fidelity: {args.resolution_HF}")
    print(f"Resolution for Low-fidelity: {args.resolution_LF}")
    print(f"Number of time steps: {args.time_steps}")
    print("#" * 50)

    return args


class KuramotoSivashinskySimulator:
    """Simulator for Kuramoto-Sivashinsky equation at different resolutions."""

    def __init__(
        self, size, length=20.0 * jnp.pi, dt=0.2, smooth=True, outer_steps=256
    ):
        """
        Initialize the KS simulator.

        Args:
            size: Grid resolution
            length: Domain length
            dt: Time step
            smooth: Whether to use smoothing
            outer_steps: Number of time steps to simulate
        """
        self.size = size
        self.length = length
        self.spatial_domain_range = (0, self.length)
        self.temporal_domain_range = (0, outer_steps * dt)
        self.dt = dt
        self.smooth = smooth
        self.outer_steps = outer_steps

        # Setup grid
        self.grid = cfd.grids.Grid((size,), domain=(self.spatial_domain_range,))
        (self.dx,) = self.grid.step
        (self.xs,) = self.grid.axes()

        # domain (x, t)
        # problem is treated as 2D with time as another dimension.
        t_eval = np.linspace(
            self.temporal_domain_range[0], self.temporal_domain_range[1], outer_steps
        )
        XX, YY = np.meshgrid(self.xs, t_eval)
        self.domain = np.concatenate(
            [XX.ravel().reshape(-1, 1), YY.ravel().reshape(-1, 1)], axis=1
        )

        # Setup time stepping
        step_fn = time_stepping.backward_forward_euler(
            spectral_equations.KuramotoSivashinsky(self.grid, smooth=self.smooth),
            self.dt,
        )
        self.rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, self.outer_steps))

    def generate_initial_conditions(self, num_trajectories=10000, seed=37723):
        """
        Generate initial conditions using sine waves.

        Args:
            num_trajectories: Number of trajectories to generate
            seed: Random seed

        Returns:
            Array of shape (num_trajectories, size) with initial conditions
        """
        key = jax.random.key(seed)
        u0 = 0.0

        for i in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
            subkey1, subkey2, key = jax.random.split(key, 3)
            u0 += jax.random.uniform(subkey1, (num_trajectories, 1)) * jnp.sin(
                (2 * i * jnp.pi / self.length) * self.xs
                + self.length
                / self.size
                * jax.random.uniform(subkey2, (num_trajectories, 1))
            )

        return jax.lax.stop_gradient(u0.reshape(*u0.shape))

    def simulate_trajectories(self, initial_conditions):
        """
        Simulate trajectories from given initial conditions.

        Args:
            initial_conditions: Array of shape (num_trajectories, size)

        Returns:
            Array of shape (num_trajectories, outer_steps, size) with trajectories
        """
        real_space_trajectory = []

        for ii, v0 in enumerate(initial_conditions):
            if ii % 10 == 0:
                print(f"sample: {ii} / {len(initial_conditions)}")
            v0_spectral = jnp.fft.rfft(v0)
            _, trajectory = jax.device_get(self.rollout_fn(v0_spectral))
            real_space_trajectory.append(jnp.fft.irfft(trajectory).real)

        return jnp.array(real_space_trajectory)

    def filter_nan_trajectories(self, trajectories, max_trajectories=10000):
        """
        Filter out trajectories containing NaN values.

        Args:
            trajectories: Array of trajectories
            max_trajectories: Maximum number of trajectories to keep

        Returns:
            Filtered trajectories
        """
        nan_indices = jnp.unique(jnp.argwhere(jnp.isnan(trajectories))[:, 0])
        mask = jnp.ones(trajectories.shape[0], dtype=bool)

        if nan_indices.shape[0] != 0:
            mask = mask.at[nan_indices].set(False)

        filtered = trajectories[mask]
        return filtered[:max_trajectories]

    def run(self, num_trajectories=10000, seed=37723):
        """
        Run complete simulation pipeline.

        Args:
            num_trajectories: Number of trajectories to generate
            seed: Random seed

        Returns:
            Filtered trajectories
        """
        initial_conditions = self.generate_initial_conditions(num_trajectories, seed)
        trajectories = self.simulate_trajectories(initial_conditions)
        return self.filter_nan_trajectories(trajectories, num_trajectories)


class MultiFidelityKSComparison:
    """Compare Kuramoto-Sivashinsky simulations at different resolutions."""

    def __init__(self, low_res=48, high_res=256, **kwargs):
        """
        Initialize multi-fidelity comparison.

        Args:
            low_res: Low fidelity resolution
            high_res: High fidelity resolution
            **kwargs: Additional arguments for simulators
        """
        self.low_res = low_res
        self.high_res = high_res

        self.high_sim = KuramotoSivashinskySimulator(high_res, **kwargs)
        self.low_sim = KuramotoSivashinskySimulator(low_res, **kwargs)

        self.spatial_domain_range = self.high_sim.spatial_domain_range
        self.temporal_domain_range = self.high_sim.temporal_domain_range

    def filter_invalid_samples(self, high_traj, low_traj):
        """
        Remove samples with NaN or Inf values in either high_traj or low_traj.

        Args:
            high_traj (np.ndarray): Array of shape (B, C, *dims)
            low_traj (np.ndarray): Array of shape (B, C, *dims)
        Returns:
            (np.ndarray, np.ndarray): Filtered high_traj and low_traj
        """
        assert high_traj.shape[0] == low_traj.shape[0], "Batch size mismatch"

        # Check validity for each sample along the batch dimension
        valid_high = np.isfinite(high_traj).all(axis=tuple(range(1, high_traj.ndim)))
        valid_low = np.isfinite(low_traj).all(axis=tuple(range(1, low_traj.ndim)))

        # Combined mask: sample is valid only if both high and low are finite
        valid_mask = np.logical_and(valid_high, valid_low)
        num_valid_samples = valid_mask.sum()

        print(
            f"Number of valid samples (not Nan/Inf):"
            f"{num_valid_samples} / {high_traj.shape[0]}"
        )

        # Filter along batch axis
        return high_traj[valid_mask], low_traj[valid_mask], valid_mask

    def run_comparison(self, num_trajectories=10000, seed=37723):
        """
        Run simulations at both resolutions and compare.

        Args:
            num_trajectories: Number of trajectories
            seed: Random seed

        Returns:
            Dictionary with high_res, low_res, and upsampled_low_res trajectories
        """
        # Generate initial conditions at high resolution
        high_init = self.high_sim.generate_initial_conditions(num_trajectories, seed)

        # Downsample for low resolution
        low_init = image.resize(high_init, (num_trajectories, self.low_res), "bicubic")

        # Run simulations
        print(f"Running high resolution ({self.high_res}) simulation...")
        high_traj = self.high_sim.simulate_trajectories(high_init)

        print(f"Running low resolution ({self.low_res}) simulation...")
        low_traj = self.low_sim.simulate_trajectories(low_init)

        # filter invalid samples
        high_traj_filtered, low_traj_filtered, valid_mask = self.filter_invalid_samples(
            high_traj, low_traj
        )
        num_valid_samples = len(high_traj_filtered)

        # filter invalid initial conditions
        high_init_filtered = high_init[valid_mask]
        low_init_filtered = low_init[valid_mask]
        upsampled_low_init_filtered = image.resize(
            low_init_filtered, (valid_mask.sum(), self.high_res), "bicubic"
        )

        # Upsample low resolution to match high resolution
        upsampled_low = image.resize(
            low_traj_filtered,
            (num_valid_samples, self.high_sim.outer_steps, self.high_res),
            "bicubic",
        )

        return {
            "high_domain": self.high_sim.domain,
            "high_init": high_init_filtered,
            "high_res": high_traj_filtered,
            "low_domain": self.low_sim.domain,
            "low_init": low_init_filtered,
            "upsampled_low_init": upsampled_low_init_filtered,
            "low_res": low_traj_filtered,
            "upsampled_low_res": upsampled_low,
        }

    def compare_energy_spectra(self, traj_hf, traj_lf, plot=True, plot_idx=0):
        """
        Compare energy spectra and total energy between HF and LF KS trajectories.

        Args:
            traj_hf: Array, shape (num_traj, Nt, Nx)
                High-fidelity trajectory (time vs space).
            traj_lf: Array, shape (num_traj, Nt, Nx)
                Low-fidelity trajectory (time vs space).
            plot: bool, optional
                If True, plots the energy spectra.
            plot_idx: int
                Index of trajectory to analyze.

        Returns:
            dict with energy spectra and total energies for both resolutions.
        """
        assert (plot_idx <= len(traj_hf)) and (plot_idx <= len(traj_lf))
        # Convert to numpy for FFT operations
        uh = np.array(traj_hf[plot_idx])
        ul = np.array(traj_lf[plot_idx])
        Nt_h, Nx_h = uh.shape
        Nt_l, Nx_l = ul.shape

        # Define wavenumbers
        k_h = np.fft.fftfreq(Nx_h, d=self.high_sim.length / Nx_h) * 2 * np.pi
        k_l = np.fft.fftfreq(Nx_l, d=self.low_sim.length / Nx_l) * 2 * np.pi

        # Compute FFT in space and energy spectra
        Uh = np.fft.fft(uh, axis=1)
        Ul = np.fft.fft(ul, axis=1)

        Eh = np.mean(np.abs(Uh) ** 2, axis=0) / Nx_h
        El = np.mean(np.abs(Ul) ** 2, axis=0) / Nx_l

        # Only keep nonnegative frequencies for plotting
        half_h = Nx_h // 2
        half_l = Nx_l // 2
        k_h_pos = k_h[:half_h]
        k_l_pos = k_l[:half_l]
        Eh_pos = Eh[:half_h]
        El_pos = El[:half_l]

        # Compute total energy (Parseval)
        E_total_h = np.sum(Eh_pos)
        E_total_l = np.sum(El_pos)

        if plot:
            plt.figure(figsize=(6, 4))
            plt.loglog(k_h_pos, Eh_pos, label=r"High-fidelity", color="k")
            plt.loglog(k_l_pos, El_pos, label=r"Low-fidelity", color="grey", alpha=0.6)
            plt.xlabel("Wavenumber, $k$")
            plt.ylabel("Energy spectrum, $E(k)$")
            plt.title("Time-averaged energy spectrum")
            plt.legend()
            plt.tight_layout()
            plt.savefig("energy_spectra.png")
            plt.close()

            print(f"Total energy (HF): {E_total_h:.3e}")
            print(f"Total energy (LF): {E_total_l:.3e}")
            print(f"Energy ratio (LF/HF): {E_total_l/E_total_h:.3f}")

        return {
            "k_h": k_h_pos,
            "E_h": Eh_pos,
            "k_l": k_l_pos,
            "E_l": El_pos,
            "E_total_h": E_total_h,
            "E_total_l": E_total_l,
        }

    def make_field_plot(self, results, trajectory_idx=0, vmin=-3, vmax=3):
        """
        Plot comparison of high and low resolution simulations.

        Args:
            results: Dictionary from run_comparison
            trajectory_idx: Which trajectory to plot
            vmin, vmax: Color scale limits
        """
        fig, axs = plt.subplots(
            1,
            3,
            figsize=(15, 5),
            dpi=300,
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )

        assert (trajectory_idx <= len(results["high_res"])) and (
            trajectory_idx <= len(results["upsampled_low_res"])
        )

        high_res_data = results["high_res"][trajectory_idx]  # (high_res, high_res)
        upsampled_low_data = results["upsampled_low_res"][
            trajectory_idx
        ]  # (high_res, high_res)

        im0 = axs[0].imshow(
            upsampled_low_data,
            vmax=vmax,
            vmin=vmin,
            aspect="auto",
            interpolation="bicubic",
            extent=(
                self.spatial_domain_range[0],
                self.spatial_domain_range[1],
                self.temporal_domain_range[1],
                self.temporal_domain_range[0],
            ),
        )
        axs[0].set_title(r"Low-fidelity")

        axs[1].imshow(
            high_res_data,
            vmax=vmax,
            vmin=vmin,
            aspect="auto",
            interpolation="bicubic",
            extent=(
                self.spatial_domain_range[0],
                self.spatial_domain_range[1],
                self.temporal_domain_range[1],
                self.temporal_domain_range[0],
            ),
        )
        axs[1].set_title(r"High-fidelity")

        diff = jnp.abs(upsampled_low_data - high_res_data)
        axs[2].imshow(
            diff,
            vmax=vmax,
            vmin=vmin,
            aspect="auto",
            interpolation="bicubic",
            extent=(
                self.spatial_domain_range[0],
                self.spatial_domain_range[1],
                self.temporal_domain_range[1],
                self.temporal_domain_range[0],
            ),
        )
        axs[2].set_title("Absolute Error")

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$t$")
            ax.label_outer()

        fig.colorbar(im0, ax=axs[-1], orientation="vertical", pad=0.1, fraction=0.046)
        plt.savefig("data_snapshot.png")
        plt.close()

    def make_joint_plot(self, traj_hf, traj_lf, plot_idx=0):
        """
        Plot joint distribution between the low and high fidelity

        Args:
            results: Dictionary from run_comparison
            trajectory_idx: Which trajectory to plot
            vmin, vmax: Color scale limits
        """
        assert (plot_idx <= len(traj_hf)) and (plot_idx <= len(traj_lf))
        traj_hf_plot = traj_hf[plot_idx].flatten()
        traj_lf_plot = traj_lf[plot_idx].flatten()

        def _plot(x, y, ax):
            assert x.shape == y.shape, "x and y must have the same shape"
            r, _ = pearsonr(x, y)

            # Create scatter plot with KDE coloring
            xy = np.vstack([x, y])
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

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300, constrained_layout=True)
        _plot(traj_lf_plot, traj_hf_plot, ax)
        ax.set_xlabel(r"Low-fidelity")
        ax.set_ylabel(r"High-fidelity")
        plt.savefig("joint_plot.png")
        plt.close()

    def compute_average_pearson(self, traj_hf, traj_lf):
        assert traj_hf.shape == traj_lf.shape

        def _comp_pearson(x, y):
            assert x.shape == y.shape
            r, _ = pearsonr(x, y)
            return r

        r_list = []
        n_samples = len(traj_hf)
        for ii in range(n_samples):
            r_list.append(_comp_pearson(traj_lf[ii].flatten(), traj_hf[ii].flatten()))
        avg_r = np.mean(np.array(r_list))
        print(
            f"{n_samples} sample average Pearson correlation coefficient: {avg_r: .4f}"
        )

    def full_analysis(self, num_trajectories=10000, seed=37723, trajectory_idx=0):
        """
        Run complete analysis including simulation and energy spectra comparison.

        Args:
            num_trajectories: Number of trajectories
            seed: Random seed
            trajectory_idx: Which trajectory to analyze

        Returns:
            Dictionary with simulation results and energy spectra
        """
        # Run simulations
        results = self.run_comparison(num_trajectories, seed)

        # Plot trajectory comparison
        print("\n=== Trajectory Comparison ===")
        self.make_field_plot(results, trajectory_idx=trajectory_idx)

        # Compare energy spectra
        print("\n=== Energy Spectra Comparison ===")
        spectra = self.compare_energy_spectra(
            results["high_res"], results["low_res"], plot=True, plot_idx=trajectory_idx
        )

        results["energy_spectra"] = spectra

        # Compare energy spectra
        print("\n=== Joint Distribution Comparison ===")
        self.make_joint_plot(
            results["high_res"], results["upsampled_low_res"], plot_idx=trajectory_idx
        )
        self.compute_average_pearson(results["high_res"], results["upsampled_low_res"])

        return results

    def create_data_dict(self, results):
        # high-fidelity
        hf_solution = np.expand_dims(results["high_res"], 1)
        Nt = hf_solution.shape[-1]  # time resolution
        # Nx = hf_solution.shape[-2]  # space resolution
        # low-fidelity
        lf_solution = np.expand_dims(results["upsampled_low_res"], 1)
        # high-fidleity domain
        hf_domain = results["high_domain"]
        # low-fidleity domain (upsampled)
        lf_domain = results["high_domain"]
        # condition
        condition = np.expand_dims(
            np.tile(np.expand_dims(results["high_init"], 1), (1, Nt, 1)), 1
        )
        # condition domain (same as the field domain)
        condition_domain = results["high_domain"]
        print(f"High-fidelity solution shape: {hf_solution.shape}")
        print(f"Low-fidelity solution shape: {lf_solution.shape}")
        print(f"High-fidelity domain shape: {hf_domain.shape}")
        print(f"Low-fidelity domain shape: {lf_domain.shape}")
        print(f"Condition shape: {condition.shape}")
        print(f"Condition domain shape: {condition_domain.shape}")

        high_data = {
            "field": hf_solution,
            "condition": condition,
            "field_domain": hf_domain,
            "condition_domain": condition_domain,
        }

        low_data = {
            "field": lf_solution,
            "condition": condition,
            "field_domain": lf_domain,
            "condition_domain": condition_domain,
        }

        np.savez("high_data.npz", **high_data)
        np.savez("low_data.npz", **low_data)


if __name__ == "__main__":
    # parse the args
    args = parse_args()
    # multi-fidelity
    mf = MultiFidelityKSComparison(
        low_res=args.resolution_LF,
        high_res=args.resolution_HF,
        outer_steps=args.time_steps,
    )
    # perform the analysis
    results = mf.full_analysis(
        num_trajectories=args.n_samples, seed=37723, trajectory_idx=2  # for plotting
    )
    mf.create_data_dict(results)
