# examples/darcy/gen_data.py
"""
Darcy Flow problem with random permeabiltiy field.
Modified from: https://github.com/christian-jacobsen/CoCoGen/tree/master
for multi-fidelity training.
Notes:
    1. For default config, average times are
        (a) High-fidelity: 1.49e-2
        (a) Low-fidelity: 6.04e-4,
    resulting in a cost savings of nearly 25 times.
"""

import numpy as np
import math
import scipy
import argparse
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import pearsonr, gaussian_kde
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator

plt.style.use("../../scripts/journal.mplstyle")


def parse_args():
    """Parse command line arguments for data generation."""

    parser = argparse.ArgumentParser(
        prog="Darcy",
        description="generate synthetic data for Darcy flow",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=11000,
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
        default=16,
        help="Number of discretization points for the low-fidelity model",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of cores to use during generation",
    )

    parser.add_argument(
        "--ntheta_HF",
        type=int,
        default=128,
        help="Parammeterization dimension of permeability field",
    )

    parser.add_argument(
        "--ntheta_LF",
        type=int,
        default=64,
        help="Parameterization dimension of permeability field",
    )

    args = parser.parse_args()
    print("#" * 50)
    print("Darcy Flow")
    print("#" * 50)
    print(f"Number of samples: {args.n_samples}")
    print(f"Number of threads: {args.threads}")
    print(f"High-fidelity parameterization dimension: {args.ntheta_HF}")
    print(f"High-fidelity resolution: {args.resolution_HF}")
    print(f"Low-fidelity parameterization dimension: {args.ntheta_LF}")
    print(f"Low-fidelity resolution: {args.resolution_LF}")
    print("#" * 50)
    return args


def check_blowup(field: np.ndarray, string: str = ""):
    """check if the field has blown up (nan or inf values)"""
    if np.isnan(field).any() or np.isinf(field).any():
        raise ValueError(f"{string} field has blown up!")


class Darcysolver:
    def __init__(self, resolution: int, ntheta: int = 128, threads: int = 32):
        self.resolution = resolution
        self.ntheta = ntheta  # mode truncations
        self.threads = threads  # number of threads for parallelization
        # mesh
        self.xv = np.linspace(0, 1, self.resolution)
        XX, YY = np.meshgrid(self.xv, self.xv, indexing="ij")
        self.mesh = np.concatenate((XX.reshape(-1, 1), YY.reshape(-1, 1)), axis=1)
        self.domain = self.mesh.reshape(self.resolution, self.resolution, 2).transpose(
            2, 0, 1
        )
        # correlation matrix params
        self.corr_params = {"c0": 0.1, "sigma": 1.0, "thresh": 1e-12}  # original
        # source
        # self.source_strength = 10.0 # original
        self.source_strength = 50.0  # modified
        # A_matrix for darcy flow (with integral enforcement)
        self.A_reg = self._get_A_matrix()

    def _get_A_matrix(self):
        """get the A matrix for the darcy flow with integral enforcement"""
        x = np.linspace(0, 1, self.resolution)
        dx = x[1] - x[0]
        nx = len(x)
        A = np.zeros((1, nx**2))
        j = 0
        i = 0
        for k in range(nx**2):
            if np.mod(k, nx) == 0:
                j += 1
                i = 1

            if (i == 1) and (j == 1):  # bottom left corner
                A[0, k] = 1
            elif (i == nx) and (j == nx):  # top right corner
                A[0, k] = 1
            elif (i == 1) and (j == nx):  # top left corner
                A[0, k] = 1
            elif (i == nx) and (j == 1):  # bottom right corner
                A[0, k] = 1
            elif i == 1:  # left boundary
                A[0, k] = 2
            elif j == 1:  # bottom boundary
                A[0, k] = 2
            elif i == nx:  # right boundary
                A[0, k] = 2
            elif j == nx:  # top boundary
                A[0, k] = 2
            else:  # interior
                A[0, k] = 4

        return A * dx**2 / 4

    def _get_correlation_matrix(self):
        """get the correlation matrix"""
        n = self.mesh.shape[0]
        c0 = self.corr_params["c0"] * np.ones((self.mesh.shape[1], 1))
        c0 = 1 / (c0**2)
        sigma = self.corr_params["sigma"]
        C = np.zeros((n, n))
        for i in range(n):
            # gaussian correlation
            point = self.mesh[i : i + 1, :]
            X = (self.mesh - point * np.ones((n, 2))) ** 2
            C[:, i] = sigma * np.exp(-np.sqrt(np.matmul(X, c0)))[:, 0]

        thresh = self.corr_params["thresh"]
        C[C < thresh] = 0

        return C

    def _get_random_field_lu(self):
        """compute the eigenvalues and eigenvectors of the covariance matrix"""
        # covariance matrix
        C = self._get_correlation_matrix()
        # compute the eigenvalues and eigenvectors
        S, U = scipy.sparse.linalg.eigs(C, k=self.ntheta)
        S_t, U_t = scipy.sparse.linalg.eigs(C, k=self.ntheta + 10)
        ev = np.abs(np.real(S))
        inds = np.flip(np.argsort(ev))
        ev = ev[inds]
        U = np.real(U)[:, inds]
        assert (
            ev.ndim == 1 and ev.shape[0] == self.ntheta
        ), "Eigen values dimension mismatch"
        assert (
            U.ndim == 2
            and U.shape[0] == self.resolution**2
            and U.shape[1] == self.ntheta
        ), "Eigen vectors dimension mismatch"
        return ev, U

    def _get_generative_params(self, n_samples):
        """get the generative parameters"""
        theta = np.random.rand(self.ntheta, n_samples) * 2.5
        # theta = np.ones((self.ntheta, n_samples)) * 2.5 # for testing purposes
        return theta

    def _sample_permeability(self, n_samples):
        """sample the permeabiilty field"""
        # get the eigen values and eigen vector os the covariance matrix
        ev, U = self._get_random_field_lu()
        # get the generative parameters
        theta = self._get_generative_params(n_samples)
        # construct permeability field (B, resolution, resolution)
        K = self._construct_permeability_from_UL(ev, U, theta)

        permeability_data = {
            "permeability": K,
            "generative_params": theta,
            "eigenvalues": ev,
            "eigenvectors": U,
        }
        return permeability_data

    def _construct_permeability_from_UL(
        self, ev: np.ndarray, U: np.ndarray, theta: np.ndarray
    ):
        """construct permeability field from eigen values,
        eigen vectors and generative parameters"""
        resolution = int(math.sqrt(U.shape[0]))
        assert resolution**2 == U.shape[0], "Resolution mismatch"
        assert (
            theta.ndim == 2 and theta.shape[0] == self.ntheta
        ), "Generative parameters dimension mismatch"
        assert (
            ev.ndim == 1 and ev.shape[0] == self.ntheta
        ), "Eigen values dimension mismatch"
        assert (
            U.ndim == 2 and U.shape[0] == resolution**2 and U.shape[1] == self.ntheta
        ), "Eigen vectors dimension mismatch"
        UL = U * np.sqrt(ev[np.newaxis, :])
        n_samples = theta.shape[1]
        K = np.exp(UL @ theta).T.reshape(n_samples, resolution, resolution)
        return K

    def grad_K(self, K, i, j, dx, dim):
        """compute gradient of K at location (i,j) in dimension dim"""
        dK = 0
        if dim == 1:
            dK = (K[i + 1, j] - K[i - 1, j]) / (2 * dx)
        elif dim == 2:
            dK = (K[i, j + 1] - K[i, j - 1]) / (2 * dx)
        return dK

    def form_matrix(self, K, xv, dx):
        """Form the matrix A and the source vector f for the Darcy flow problem."""
        nx = len(xv)
        A = np.zeros((nx**2, nx**2))
        f = np.zeros((nx**2, 1))

        j = -1
        i = 0

        dx2 = dx**2
        for k in range(nx**2):
            if (np.mod(k, nx)) == 0:
                j += 1
                i = 0

            A[k, k] = K[i, j] * 4 / (dx2)

            # corners
            if (i == 0) and (j == 0):
                A[k, k + 1] = -2 * K[i, j] / (dx2)
                A[k, k + nx] = -2 * K[i, j] / (dx2)
            elif (i == (nx - 1)) and (j == (nx - 1)):
                A[k, k - 1] = -2 * K[i, j] / (dx2)
                A[k, k - nx] = -2 * K[i, j] / (dx2)
            elif (i == 0) and (j == (nx - 1)):
                A[k, k + 1] = -2 * K[i, j] / (dx2)
                A[k, k - nx] = -2 * K[i, j] / (dx2)
            elif (i == (nx - 1)) and (j == 0):
                A[k, k - 1] = -2 * K[i, j] / (dx2)
                A[k, k + nx] = -2 * K[i, j] / (dx2)

            # bondaries
            elif (i == 0) or (j == 0) or (i == (nx - 1)) or (j == (nx - 1)):
                if i == 0:
                    gK = self.grad_K(K, i, j, dx, 2) / (2 * dx)
                    A[k, k + 1] = -2 * K[i, j] / dx2
                    A[k, k + nx] = -K[i, j] / dx2 - gK
                    A[k, k - nx] = -K[i, j] / dx2 + gK
                elif i == (nx - 1):
                    gK = self.grad_K(K, i, j, dx, 2) / (2 * dx)
                    A[k, k - 1] = -2 * K[i, j] / dx2
                    A[k, k + nx] = -K[i, j] / dx2 - gK
                    A[k, k - nx] = -K[i, j] / dx2 + gK
                elif j == 0:
                    gK = self.grad_K(K, i, j, dx, 1) / (2 * dx)
                    A[k, k + nx] = -2 * K[i, j] / dx2
                    A[k, k - 1] = -K[i, j] / dx2 + gK
                    A[k, k + 1] = -K[i, j] / dx2 - gK
                elif j == (nx - 1):
                    gK = self.grad_K(K, i, j, dx, 1) / (2 * dx)
                    A[k, k - nx] = -2 * K[i, j] / dx2
                    A[k, k - 1] = -K[i, j] / dx2 + gK
                    A[k, k + 1] = -K[i, j] / dx2 - gK

            # interior
            else:
                gK1 = self.grad_K(K, i, j, dx, 1) / (2 * dx)
                gK2 = self.grad_K(K, i, j, dx, 2) / (2 * dx)
                fac = -K[i, j] / dx2
                A[k, k - 1] = fac + gK1
                A[k, k + 1] = fac - gK1
                A[k, k + nx] = fac - gK2
                A[k, k - nx] = fac + gK2

            x, y = xv[i], xv[j]

            # source function
            if (np.abs(x - 0.0625) <= 0.0625) and (np.abs(y - 0.0625) <= 0.0625):
                f[k, 0] = self.source_strength
            elif (np.abs(x - 1 + 0.0625) <= 0.0625) and (
                np.abs(y - 1 + 0.0625) <= 0.0625
            ):
                f[k, 0] = -self.source_strength

            i += 1
        return A, f

    def compute_u(self, P, K, nx, xv, dx):
        U1, U2 = np.zeros(P.shape), np.zeros(P.shape)
        for i in range(nx):
            for j in range(nx):
                if (
                    ((j == 0) or (j == (nx - 1))) and (i != 0) and (i != (nx - 1))
                ):  # bottom or top boundary (no corners)
                    U1[i, j] = -K[i, j] * (P[i + 1, j] - P[i - 1, j]) / (2 * dx)
                elif (
                    ((i == 0) or (i == (nx - 1))) and (j != 0) and (j != (nx - 1))
                ):  # left or right boundary
                    U2[i, j] = -K[i, j] * (P[i, j + 1] - P[i, j - 1]) / (2 * dx)
                elif (
                    (i == 0 and j == 0)
                    or (i == (nx - 1) and j == (nx - 1))
                    or (i == 0 and j == (nx - 1))
                    or (j == 0 and i == (nx - 1))
                ):  # corners
                    U1[i, j] = 0
                    U2[i, j] = 0
                else:  # interior
                    U1[i, j] = -K[i, j] * (P[i + 1, j] - P[i - 1, j]) / (2 * dx)
                    U2[i, j] = -K[i, j] * (P[i, j + 1] - P[i, j - 1]) / (2 * dx)

        return U1, U2

    def _solve(self, K: np.ndarray, sample_index: int, sparse_solve: bool = True):
        if sample_index % 1000 == 0:
            print(f"Generating sample {sample_index}...")
        xv = np.linspace(0, 1, self.resolution)
        dx = xv[1] - xv[0]
        A, f = self.form_matrix(K, xv, dx)
        if sparse_solve:
            # Form KKT matrix
            n = A.shape[0]
            KKT = np.zeros((n + 1, n + 1))
            KKT[:n, :n] = A
            KKT[:n, n] = self.A_reg.flatten()  # or self.A_reg.ravel()
            KKT[n, :n] = self.A_reg.T.flatten()

            # Form KKT right-hand side
            rhs = np.zeros((n + 1, 1))
            rhs[:n] = f
            KKT_sparse = csr_matrix(KKT)
            solution = spsolve(KKT_sparse, rhs.flatten())
            P = solution[:n]
            P = P.reshape((self.resolution, self.resolution))
        else:
            A = np.concatenate((A, self.A_reg), axis=0)
            f = np.concatenate((f, np.zeros((1, 1))), axis=0)
            P, _, _, _ = scipy.linalg.lstsq(A, f)
            P = P.reshape((self.resolution, self.resolution))
        U1, U2 = self.compute_u(P, K, self.resolution, xv, dx)

        return P, U1, U2

    def solve(self, K: np.ndarray):
        """solve the darcy flow problem"""
        tic = time.time()
        n_samples = K.shape[0]
        results = Parallel(n_jobs=self.threads)(
            delayed(self._solve)(K[i], i) for i in range(0, n_samples)
        )
        P, U1, U2 = zip(*results)
        P = np.stack(P)  # high-fidelity pressure field
        U1 = np.stack(U1)  # high-fidelity velocity field in x-direction
        U2 = np.stack(U2)  # high-fidelity velocity field in y-direction
        # data dict
        data_dict = {
            "pressure": P,
            "velocity_x": U1,
            "velocity_y": U2,
        }
        toc = time.time()
        avg_time = (toc - tic) / n_samples
        print(f"Total solve time for {n_samples} samples: {toc - tic: .2e} seconds")
        print(f"Average solve time per sample: {avg_time: .2e} seconds")
        return data_dict


class MultiFidelity:
    def __init__(
        self,
        ntheta_HF: int = 128,
        ntheta_LF: int = 32,
        resolution_HF: int = 64,
        resolution_LF: int = 32,
        n_samples: int = 1,
        threads: int = 32,
    ):
        self.ntheta_HF = ntheta_HF
        self.ntheta_LF = ntheta_LF
        self.resolution_HF = resolution_HF
        self.resolution_LF = resolution_LF
        self.n_samples = n_samples

        assert (
            self.ntheta_LF <= self.ntheta_HF
        ), "Low-fidelity parameterization dimension must be less than high-fidelity"
        assert (
            self.resolution_LF <= self.resolution_HF
        ), "Low-fidelity resolution must be less than or equal to high-fidelity"

        # high-fidelity solver
        self.solver_HF = Darcysolver(
            resolution=self.resolution_HF, ntheta=self.ntheta_HF, threads=threads
        )
        # low-fidelity solver
        self.solver_LF = Darcysolver(
            resolution=self.resolution_LF, ntheta=self.ntheta_LF, threads=threads
        )

    def _get_permeability_fields(self):
        """get the permeability fields for HF and LF"""
        # get the high-fidelity permeability fields
        print("generating permeability fields for high-fidelity...")
        permeability_HF_data = self.solver_HF._sample_permeability(self.n_samples)
        K_HF = permeability_HF_data["permeability"]
        # truncate for low-fidelity
        print("generating permeability fields for low-fidelity...")
        ev_LF = permeability_HF_data["eigenvalues"][: self.ntheta_LF]
        U_LF = permeability_HF_data["eigenvectors"][:, : self.ntheta_LF]
        theta_LF = permeability_HF_data["generative_params"][: self.ntheta_LF, :]
        # construct low-fidelity permeability fields (B, resolution_HF, resolution_HF)
        K_LF = self.solver_LF._construct_permeability_from_UL(ev_LF, U_LF, theta_LF)
        # restric to the low-fidelity resolution (B, resolution_LF, resolution_LF)
        K_LF_int = np.zeros((self.n_samples, self.resolution_LF, self.resolution_LF))
        print("interpolating low-fidelity permeability fields to low-fidelity mesh...")
        pbar = tqdm(range(self.n_samples), desc="LF permeability interpolation")
        for ii in pbar:
            interpolator = RegularGridInterpolator(
                (self.solver_HF.xv, self.solver_HF.xv),
                K_LF[ii],
                method="linear",
            )
            K_LF_int[ii] = interpolator(self.solver_LF.mesh.reshape(-1, 2)).reshape(
                self.resolution_LF, self.resolution_LF
            )
        pbar.close()
        self._make_sample_comparison_plot(
            K_LF, K_HF, field_name="permeability", n_samples=10
        )
        return K_HF, K_LF, K_LF_int

    def _make_joint_plot(
        self, data_dict_HF: dict, data_dict_LF: dict, random_index: bool = False
    ):
        if random_index:
            sample_index = np.random.randint(0, self.n_samples)
        else:
            sample_index = 0
        print(f"Making joint plot for sample index {sample_index}")

        # extract fields
        K_HF = data_dict_HF["permeability"][sample_index].flatten()
        P_HF = data_dict_HF["pressure"][sample_index].flatten()
        K_LF = data_dict_LF["permeability"][sample_index].flatten()
        P_LF = data_dict_LF["pressure"][sample_index].flatten()

        def _plot(x, y, ax):
            assert (
                x.shape == y.shape
            ), f"x ({x.shape}) and y ({y.shape}) must have the same shape"
            r, _ = pearsonr(x, y)

            # Create scatter plot with KDE coloring
            xy = np.vstack([x, y])
            xy = xy + 1e-6 * np.random.randn(*xy.shape)
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

        fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=300, layout="compressed")
        _plot(K_LF, K_HF, axs[0])
        axs[0].set_title(r"Permeability, $K$")
        _plot(P_LF, P_HF, axs[1])
        axs[1].set_title(r"Pressure, $P$")

        for ii, ax in enumerate(axs):
            ax.set_xlabel(r"Low-fidelity")
            if ii == 0:
                ax.set_ylabel(r"High-fidelity")

        plt.savefig("joint_plot.png", dpi=300)
        plt.close()

    def _make_sample_comparison_plot(
        self, samples_LF, samples_HF, field_name: str, n_samples=10
    ):
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
            # vmin = min(samples_LF[ii].min(), samples_HF[ii].min())
            # vmax = max(samples_LF[ii].max(), samples_HF[ii].max())
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
            ax.set_xlabel(r"$x_{1}$")
            ax.set_ylabel(r"$x_{2}$")
            ax.label_outer()
        plt.savefig(f"{field_name}_sample_comparison.png", dpi=300)

    def _make_field_plot(
        self, data_dict_HF: dict, data_dict_LF: dict, random_index: bool = False
    ):
        if random_index:
            sample_index = np.random.randint(0, self.n_samples)
        else:
            sample_index = 0
        print(f"Making field plot for sample index {sample_index}")

        # extract fields
        K_HF = data_dict_HF["permeability"][sample_index]
        P_HF = data_dict_HF["pressure"][sample_index]
        K_LF = data_dict_LF["permeability"][sample_index]
        P_LF = data_dict_LF["pressure"][sample_index]
        assert (
            K_HF.shape == K_LF.shape
        ), "HF and LF permeability fields must have the same shape for plotting"
        assert (
            P_HF.shape == P_LF.shape
        ), "HF and LF pressure fields must have the same shape for plotting"
        # plot
        fig, axs = plt.subplots(
            2,
            3,
            figsize=(12, 6),
            dpi=300,
            layout="compressed",
            sharex=True,
            sharey=True,
        )
        # permeability plot
        vmin_K = min(K_LF.min(), K_HF.min())
        vmax_K = max(K_LF.max(), K_HF.max())
        im_K = axs[0, 0].imshow(
            K_LF,
            aspect="equal",
            interpolation="bicubic",
            vmin=vmin_K,
            vmax=vmax_K,
            origin="lower",
        )
        axs[0, 1].imshow(
            K_HF,
            aspect="equal",
            interpolation="bicubic",
            vmin=vmin_K,
            vmax=vmax_K,
            origin="lower",
        )
        axs[0, 0].set_title(r"Low-fidelity")
        axs[0, 1].set_title(r"High-fidelity")
        cbar_K = fig.colorbar(
            im_K,
            ax=axs[0, 1],
            orientation="vertical",
            fraction=0.046,
            pad=0.1,
        )
        cbar_K.set_label(r"Permeability, $K$", labelpad=15)
        cbar_K.ax.yaxis.set_label_coords(6.5, 0.5)
        # permeability difference plot
        im_Kd = axs[0, 2].imshow(
            np.abs(K_HF - K_LF),
            # aspect="equal",
            interpolation="bicubic",
            origin="lower",
        )
        axs[0, 2].set_title(r"Absolute error")
        fig.colorbar(
            im_Kd, ax=axs[0, 2], orientation="vertical", fraction=0.046, pad=0.1
        )
        # pressure plot
        vmin_P = min(P_LF.min(), P_HF.min())
        vmax_P = max(P_LF.max(), P_HF.max())
        im_p = axs[1, 0].imshow(
            P_LF,
            aspect="equal",
            interpolation="bicubic",
            vmin=vmin_P,
            vmax=vmax_P,
            origin="lower",
        )
        axs[1, 1].imshow(
            P_HF,
            aspect="equal",
            interpolation="bicubic",
            vmin=vmin_P,
            vmax=vmax_P,
            origin="lower",
        )
        cbar_P = fig.colorbar(
            im_p,
            ax=axs[1, 1],
            orientation="vertical",
            fraction=0.046,
            pad=0.1,
        )
        cbar_P.set_label(r"Pressure, $P$", labelpad=15)
        cbar_P.ax.yaxis.set_label_coords(6.5, 0.5)
        # pressure difference plot
        im_Pd = axs[1, 2].imshow(
            np.abs(P_HF - P_LF),
            # aspect="equal",
            interpolation="bicubic",
            origin="lower",
        )
        fig.colorbar(
            im_Pd, ax=axs[1, 2], orientation="vertical", fraction=0.046, pad=0.1
        )

        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r"$x_{1}$")
            ax.set_ylabel(r"$x_{2}$")
            ax.label_outer()
        fig.align_labels()
        fig.set_constrained_layout_pads(
            w_pad=0.001, h_pad=0.001, hspace=0.02, wspace=0.02
        )
        plt.savefig("data_snapshot.png", dpi=300)
        plt.close()

    def _comp_average_pearson(self, data_dict_HF: dict, data_dict_LF: dict):
        """compute the average pearson correlation coefficient"""

        def _comp_pearson(x: np.ndarray, y: np.ndarray):
            assert x.shape == y.shape, "x and y must have the same shape"
            r, _ = pearsonr(x, y)
            return r

        r_permeability_list = []
        r_pressure_list = []
        for ii in range(self.n_samples):
            # permeability
            K_HF = data_dict_HF["permeability"][ii].flatten()
            K_LF = data_dict_LF["permeability"][ii].flatten()
            r_permeability_list.append(
                _comp_pearson(
                    K_HF,
                    K_LF,
                )
            )
            # pressure
            P_HF = data_dict_HF["pressure"][ii].flatten()
            P_LF = data_dict_LF["pressure"][ii].flatten()
            r_pressure_list.append(
                _comp_pearson(
                    P_HF,
                    P_LF,
                )
            )
        assert (
            len(r_permeability_list) == len(r_pressure_list) == self.n_samples
        ), "Number of samples mismatch"

        mean_r_permeability = np.mean(np.array(r_permeability_list))
        mean_r_pressure = np.mean(np.array(r_pressure_list))

        std_r_permeability = np.std(np.array(r_permeability_list))
        std_r_pressure = np.std(np.array(r_pressure_list))

        print(
            f"{self.n_samples} sample average Pearson correlation coefficient for "
            f"permeability: {mean_r_permeability: .4f} with {std_r_permeability: .4f} "
            "standard deviation"
        )
        print(
            f"{self.n_samples} sample average Pearson correlation coefficient for "
            f"pressure: {mean_r_pressure: .4f} with {std_r_pressure: .4f} "
            "standard deviation"
        )

    def simulate(self):
        """simulate multi-fidelity data"""
        # get the permeability fields
        K_HF, K_LF, K_LF_int = self._get_permeability_fields()
        # solve high-fidelity darcy flow
        print("solving high-fidelity darcy flow...")
        data_dict_HF = self.solver_HF.solve(K_HF)
        data_dict_HF["permeability"] = K_HF
        # solve low-fidelity darcy flow
        print("solving low-fidelity darcy flow...")
        data_dict_LF = self.solver_LF.solve(K_LF_int)
        data_dict_LF["permeability"] = (
            K_LF  # original LF permeability before interpolation
        )
        # interpolate the LF solution and update the dictionary
        P_LF_int = []
        pbar = tqdm(range(self.n_samples), desc="LF pressure interpolation")
        for ii in pbar:
            # intepolate pressure
            interpolator = RegularGridInterpolator(
                (self.solver_LF.xv, self.solver_LF.xv),
                data_dict_LF["pressure"][ii],
                method="linear",
            )
            P_LF_int.append(
                interpolator(self.solver_HF.mesh.reshape(-1, 2)).reshape(
                    self.resolution_HF, self.resolution_HF
                )
            )
        pbar.close()
        P_LF_int = np.stack(P_LF_int)
        data_dict_LF["pressure"] = P_LF_int

        # compare pressure plots
        self._make_sample_comparison_plot(
            data_dict_LF["pressure"],
            data_dict_HF["pressure"],
            field_name="pressure",
            n_samples=10,
        )

        # check blow-up
        check_blowup(data_dict_HF["pressure"], "High-fidelity pressure field")
        check_blowup(data_dict_LF["pressure"], "Low-fidelity pressure field")
        check_blowup(data_dict_HF["permeability"], "High-fidelity permeability field")
        check_blowup(data_dict_LF["permeability"], "Low-fidelity permeability field")

        # check size
        assert (
            data_dict_HF["pressure"].shape == data_dict_LF["pressure"].shape
        ), "HF and LF pressure fields must have the same shape after interpolation"
        assert (
            data_dict_HF["permeability"].shape == data_dict_LF["permeability"].shape
        ), "HF and LF permeability fields must have the same shape after interpolation"
        assert (
            len(data_dict_HF["pressure"]) == self.n_samples
        ), "Number of HF pressure samples mismatch"
        assert (
            len(data_dict_LF["pressure"]) == self.n_samples
        ), "Number of LF pressure samples mismatch"
        assert (
            len(data_dict_HF["permeability"]) == self.n_samples
        ), "Number of HF permeability samples mismatch"
        assert (
            len(data_dict_LF["permeability"]) == self.n_samples
        ), "Number of LF permeability samples mismatch"

        # make field plot
        self._make_field_plot(data_dict_HF, data_dict_LF, random_index=False)
        # make joint plot
        self._make_joint_plot(data_dict_HF, data_dict_LF, random_index=False)
        # compute average pearson correlation coefficient
        self._comp_average_pearson(data_dict_HF, data_dict_LF)
        # reshape data for saving
        high_data = {
            "field": np.expand_dims(
                data_dict_HF["pressure"], 1
            ),  # (B, channel_f, *dim_f)
            "condition": np.log(
                np.expand_dims(data_dict_HF["permeability"], 1)
            ),  # (B, channel_c, *dim_c)
            "field_domain": self.solver_HF.mesh,  # flattened mesh (num_points, dim)
            "condition_domain": self.solver_HF.mesh,  # flattened mesh (num_points, dim)
        }
        low_data = {
            "field": np.expand_dims(
                data_dict_LF["pressure"], 1
            ),  # (B, channel_f, *dim_f)
            "condition": np.log(
                np.expand_dims(data_dict_LF["permeability"], 1)
            ),  # (B, channel_c, *dim_c)
            "field_domain": self.solver_HF.mesh,  # flattened mesh (num_points, dim)
            "condition_domain": self.solver_HF.mesh,  # flattened mesh (num_points, dim)
        }
        # save
        np.savez("high_fidelity.npz", **high_data)
        np.savez("low_fidelity.npz", **low_data)
        print("Data saved to high_fidelity.npz and low_fidelity.npz")
        print("Simulation complete.")


if __name__ == "__main__":
    args = parse_args()
    mf = MultiFidelity(
        ntheta_HF=args.ntheta_HF,
        ntheta_LF=args.ntheta_LF,
        resolution_HF=args.resolution_HF,
        resolution_LF=args.resolution_LF,
        n_samples=args.n_samples,
        threads=args.threads,
    )
    mf.simulate()
