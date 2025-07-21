# examples/darcy/gen_data.py
"""
Darcy Flow problem with random permeabiltiy field.
Modified from: https://github.com/christian-jacobsen/CoCoGen/tree/master
"""

import numpy as np
from joblib import Parallel, delayed
import scipy
from scipy.sparse.linalg import lsqr
from scipy.sparse import csr_matrix
from scipy.interpolate import RBFInterpolator
import argparse
import matplotlib.pyplot as plt

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
        default=4000,
        help="Number of samples to generate",
    )

    parser.add_argument(
        "-nt",
        "--n_test_samples",
        type=int,
        default=200,
        help="Number of samples to generate test",
    )

    parser.add_argument(
        "-mh",
        "--m_high",
        type=int,
        default=64,
        help="Number of discretization points for high fidelity",
    )

    parser.add_argument(
        "--ss_factor",
        type=int,
        default=2,
        help="Subsampling factor for low fidelity discretization",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of cores to use during generation",
    )

    parser.add_argument(
        "--ntheta",
        type=int,
        default=128,
        help="Parammeterization dimension of permeabiltiy field",
    )

    parser.add_argument(
        "--resume", type=int, default=0, help="Resume generation from sample i"
    )

    args = parser.parse_args()

    print(
        f"Generating {args.n_samples} samples for training and validation, "
        f"and {args.n_test_samples} for testing."
    )

    return args


def correlation_function(corr, mesh, spthresh):
    """Computes the correlation matrix C for a given mesh and correlation parameters
    Args:
        corr (dict): correlation parameters with keys "c0" and "sigma"
        mesh (np.ndarray): mesh points of shape (n, 2)
        spthresh (float): sparsity threshold for the correlation matrix
    """
    n = mesh.shape[0]
    c0 = corr["c0"] * np.ones((mesh.shape[1], 1))
    c0 = 1 / (c0**2)
    sigma = corr["sigma"]
    C = np.zeros((n, n))
    for i in range(n):
        # gaussian correlation
        point = mesh[i : i + 1, :]
        X = (mesh - point * np.ones((n, 2))) ** 2
        C[:, i] = sigma * np.exp(-np.sqrt(np.matmul(X, c0)))[:, 0]

    C[C < spthresh] = 0

    return C


def enforce_integral(resolution):
    #
    x = np.linspace(0, 1, resolution)
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


def random_field(resolution: int, ntheta: int = 128):
    """Generate the random field for the permeability.
    Args:
        resolution (int): resolution of the random field
        ntheta (int): number of eigenvalues/eigenvectors to keep
    """
    # computes the eigenvalues and eigenvectors of the covariance matrix
    xv = np.linspace(0, 1, resolution)
    corr = {"c0": 0.1, "sigma": 1.0}

    X, Y = np.meshgrid(xv, xv, indexing="ij")
    mesh = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

    C = correlation_function(corr, mesh, 1.0e-12)  # covariance matrix
    S, U = scipy.sparse.linalg.eigs(C, k=ntheta)  # eigenvalues and eigenvectors
    ev = np.abs(np.real(S))
    inds = np.flip(np.argsort(ev))
    ev = ev[inds]
    U = np.real(U)[:, inds]
    return np.diag(ev), U


def get_high_fidelity_permeability_field(
    resolution: int, n_samples: int, ntheta: int = 128
):
    """compute the high-fidelity permeability field
    Args:
        resolution (int): resolution of the permeability field
        n_samples (int): number of samples to generate
        ntheta (int): number of eigenvalues/eigenvectors to keep
    """
    # get the eigenvalues and eigenvectors of the covariance matrix
    L, U = random_field(resolution=resolution, ntheta=ntheta)
    UL = np.matmul(U, np.sqrt(L))

    K_samples = []
    W_samples = []
    for ii in range(n_samples):
        # Get the generative parameters
        W = np.random.rand(ntheta, 1) * 2.5
        W_samples.append(W)
        # Get the high-fidelity permeability field
        K = np.matmul(UL, W)
        K_samples.append(np.exp(K).reshape((resolution, resolution)))

    return np.stack(K_samples), np.stack(W_samples)


def get_low_fidelity_permeability_field(K_high: np.ndarray, ss_factor: int = 2):
    """compute the low-fidelity permeability field from the high-fidelity field
    Args:
        K_high (np.ndarray): high-fidelity permeability field of shape
                            (n_samples, resolution, resolution)
        ss_factor (int): subsampling factor for low fidelity discretization
    Returns:
        np.ndarray: low-fidelity permeability field of shape
                            (n_samples, resolution/ss_factor, resolution/ss_factor)
    """
    return K_high[:, :: args.ss_factor, :: args.ss_factor]


def grad_K(K, i, j, dx, dim):
    # gradient of K w.r.t. dimension at location
    dK = 0
    if dim == 1:
        dK = (K[i + 1, j] - K[i - 1, j]) / (2 * dx)
    elif dim == 2:
        dK = (K[i, j + 1] - K[i, j - 1]) / (2 * dx)
    return dK


def form_matrix(K, xv, dx):
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
                gK = grad_K(K, i, j, dx, 2) / (2 * dx)
                A[k, k + 1] = -2 * K[i, j] / dx2
                A[k, k + nx] = -K[i, j] / dx2 - gK
                A[k, k - nx] = -K[i, j] / dx2 + gK
            elif i == (nx - 1):
                gK = grad_K(K, i, j, dx, 2) / (2 * dx)
                A[k, k - 1] = -2 * K[i, j] / dx2
                A[k, k + nx] = -K[i, j] / dx2 - gK
                A[k, k - nx] = -K[i, j] / dx2 + gK
            elif j == 0:
                gK = grad_K(K, i, j, dx, 1) / (2 * dx)
                A[k, k + nx] = -2 * K[i, j] / dx2
                A[k, k - 1] = -K[i, j] / dx2 + gK
                A[k, k + 1] = -K[i, j] / dx2 - gK
            elif j == (nx - 1):
                gK = grad_K(K, i, j, dx, 1) / (2 * dx)
                A[k, k - nx] = -2 * K[i, j] / dx2
                A[k, k - 1] = -K[i, j] / dx2 + gK
                A[k, k + 1] = -K[i, j] / dx2 - gK

        # interior
        else:
            gK1 = grad_K(K, i, j, dx, 1) / (2 * dx)
            gK2 = grad_K(K, i, j, dx, 2) / (2 * dx)
            fac = -K[i, j] / dx2
            A[k, k - 1] = fac + gK1
            A[k, k + 1] = fac - gK1
            A[k, k + nx] = fac - gK2
            A[k, k - nx] = fac + gK2

        x, y = xv[i], xv[j]

        # source function
        if (np.abs(x - 0.0625) <= 0.0625) and (np.abs(y - 0.0625) <= 0.0625):
            f[k, 0] = 10
        elif (np.abs(x - 1 + 0.0625) <= 0.0625) and (np.abs(y - 1 + 0.0625) <= 0.0625):
            f[k, 0] = -10

        i += 1
    return A, f


def compute_u(P, K, nx, xv, dx):
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


def generate_darcy_solution(K, Areg, resolution, sample_index):
    """Solve the Darcy flow problem for a given permeability field K."""
    if sample_index % 10 == 0:
        print(f"Generating sample {sample_index}...")

    xv = np.linspace(0, 1, resolution)
    dx = xv[1] - xv[0]

    A, f = form_matrix(K, xv, dx)

    A = np.concatenate((A, Areg), axis=0)
    f = np.concatenate((f, np.zeros((1, 1))), axis=0)
    A_sparse = csr_matrix(A)
    P = lsqr(A_sparse, f.ravel(), atol=1e-5)[0]
    # P, _, _, _ = scipy.linalg.lstsq(A, f)
    P = P.reshape((resolution, resolution))

    U1, U2 = compute_u(P, K, resolution, xv, dx)

    return P, U1, U2


def interpolator(field, domain_train, domain_eval):
    """Interpolate the field using radial basis function interpolation."""
    model = RBFInterpolator(domain_train, field)
    return model(domain_eval)


def query_lf_at_hf_domain(
    K_lf, P_lf, U1_lf, U2_lf, resolution_high: int, ss_factor: int = 2
):
    """query the low-fidelity field at the high-fidelity domain
    Args:
        K_lf (np.ndarray): low-fidelity permeability field of shape
            (resolution_low, resolution_low)
        P_lf (np.ndarray): low-fidelity pressure field of shape
        (resolution_low, resolution_low)
        U1_lf (np.ndarray): low-fidelity x-velocity field of shape
                            (resolution_low, resolution_low)
        U2_lf (np.ndarray): low-fidelity y-velocity field of shape
                            (resolution_low, resolution_low)
        resolution_high (int): resolution of the high-fidelity domain
        ss_factor (int): subsampling factor for low fidelity discretization
    """
    x_high = np.linspace(0, 1, resolution_high)
    x_low = np.linspace(0, 1, resolution_high // ss_factor)
    xx_high, yy_high = np.meshgrid(x_high, x_high, indexing="ij")
    xx_low, yy_low = np.meshgrid(x_low, x_low, indexing="ij")
    domain_high = np.vstack((xx_high.ravel(), yy_high.ravel())).T
    domain_low = np.vstack((xx_low.ravel(), yy_low.ravel())).T
    # Interpolate the low-fidelity field to the high-fidelity domain using radial basis
    K_eval = interpolator(K_lf.ravel(), domain_low, domain_high)
    P_eval = interpolator(P_lf.ravel(), domain_low, domain_high)
    U1_eval = interpolator(U1_lf.ravel(), domain_low, domain_high)
    U2_eval = interpolator(U2_lf.ravel(), domain_low, domain_high)

    K_eval = K_eval.reshape((resolution_high, resolution_high))
    P_eval = P_eval.reshape((resolution_high, resolution_high))
    U1_eval = U1_eval.reshape((resolution_high, resolution_high))
    U2_eval = U2_eval.reshape((resolution_high, resolution_high))
    return K_eval, P_eval, U1_eval, U2_eval


def plot_snapshot(
    domain,
    K_high,
    K_low,
    K_recon,
    P_high,
    P_low,
    P_recon,
    resolution_high,
    ss_factor: int = 2,
):
    """Plot a snapshot of the permeability and pressure fields.
    Args:
        domain (np.ndarray): domain of shape (resolution * resolution, 2)
        K_high (np.ndarray): high-fidelity permeability field of shape
        (n_samples, resolution, resolution)
        K_low (np.ndarray): low-fidelity permeability field of shape
        (n_samples, resolution/ss_factor, resolution/ss_factor)
        K_recon (np.ndarray): reconstructed low-fidelity permeability field at
                            high-fidelity domain
        P_high (np.ndarray): high-fidelity pressure field of shape
        (n_samples, resolution, resolution)
        P_low (np.ndarray): low-fidelity pressure field of shape
        (n_samples, resolution/ss_factor, resolution/ss_factor)
        P_recon (np.ndarray): reconstructed low-fidelity pressure field at high-fidelity
                                domain
        resolution_high (int): resolution of the high-fidelity domain
        ss_factor (int): subsampling factor for low fidelity discretization
    """
    idx_plot = 0  # for reproducibility, always plot the first sample

    fig, axs = plt.subplots(
        2, 3, figsize=(12, 8), dpi=300, tight_layout=True, sharex=True, sharey=True
    )

    # Extract the plot data
    K_high_plot = K_high[idx_plot]
    K_low_plot = K_low[idx_plot]
    K_recon_plot = K_recon[idx_plot]
    P_high_plot = P_high[idx_plot]
    P_low_plot = P_low[idx_plot]
    P_recon_plot = P_recon[idx_plot]
    labels = ["High-fidelity", "Low-fidelity", "Reconstructed low-fidelity"]

    vmin_K = min(K_high_plot.min(), K_low_plot.min(), K_recon_plot.min())
    vmax_K = max(K_high_plot.max(), K_low_plot.max(), K_recon_plot.max())
    vmin_P = min(P_high_plot.min(), P_low_plot.min(), P_recon_plot.min())
    vmax_P = max(P_high_plot.max(), P_low_plot.max(), P_recon_plot.max())

    # permeability fields
    ax_K = axs[0, 0].imshow(
        K_high_plot,
        extent=(0, 1, 0, 1),
        origin="lower",
        vmin=vmin_K,
        vmax=vmax_K,
        aspect="auto",
        interpolation="bilinear",
    )
    axs[0, 1].imshow(
        K_low_plot,
        extent=(0, 1, 0, 1),
        origin="lower",
        vmin=vmin_K,
        vmax=vmax_K,
        aspect="auto",
        interpolation="bilinear",
    )
    axs[0, 2].imshow(
        K_recon_plot,
        extent=(0, 1, 0, 1),
        origin="lower",
        vmin=vmin_K,
        vmax=vmax_K,
        aspect="auto",
        interpolation="bilinear",
    )

    # pressure fields
    ax_P = axs[1, 0].imshow(
        P_high_plot,
        extent=(0, 1, 0, 1),
        origin="lower",
        vmin=vmin_P,
        vmax=vmax_P,
        aspect="auto",
        interpolation="bilinear",
    )
    axs[1, 1].imshow(
        P_low_plot,
        extent=(0, 1, 0, 1),
        origin="lower",
        vmin=vmin_P,
        vmax=vmax_P,
        aspect="auto",
        interpolation="bilinear",
    )
    axs[1, 2].imshow(
        P_recon_plot,
        extent=(0, 1, 0, 1),
        origin="lower",
        vmin=vmin_P,
        vmax=vmax_P,
        aspect="auto",
        interpolation="bilinear",
    )

    # Add colorbar for row 0 (K - permeability)
    cbar_ax_K = fig.add_axes([1, 0.56, 0.015, 0.34])  # [left, bottom, width, height]
    cbar_K = fig.colorbar(ax_K, cax=cbar_ax_K)
    cbar_K.set_label(r"$K$ (permeability)")

    # Add colorbar for row 1 (P - pressure)
    cbar_ax_P = fig.add_axes([1, 0.11, 0.015, 0.34])  # adjust to match lower row
    cbar_P = fig.colorbar(ax_P, cax=cbar_ax_P)
    cbar_P.set_label(r"$P$ (pressure)")

    for ii, ax in enumerate(axs.flatten()):
        # plot outer labels
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.label_outer()
        # add title
        if ii // 3 == 0:
            ax.set_title(labels[ii])

    plt.savefig("snapshot.png")
    plt.close()


def generate(args):
    """generate the data"""
    # total samples
    total_samples = args.n_samples + args.n_test_samples
    # Generate the permeability field
    K_high, W = get_high_fidelity_permeability_field(
        resolution=args.m_high, n_samples=total_samples, ntheta=args.ntheta
    )
    K_low = get_low_fidelity_permeability_field(K_high, args.ss_factor)
    print("Generated permeability fields...")

    # Domain
    xx_high, yy_high = np.meshgrid(
        np.linspace(0, 1, args.m_high), np.linspace(0, 1, args.m_high), indexing="ij"
    )
    domain = np.vstack(
        (xx_high.reshape(-1), yy_high.reshape(-1))
    ).T  # shape (m_high**2, 2)

    # Solve the Darcy flow problem for high-fidelity
    A_reg_high = enforce_integral(args.m_high)
    results_high = Parallel(n_jobs=args.threads)(
        delayed(generate_darcy_solution)(K_high[i], A_reg_high, args.m_high, i)
        for i in range(args.resume, total_samples)
    )
    P_high_list, U1_high_list, U2_high_list = zip(*results_high)
    P_high = np.stack(P_high_list)  # high-fidelity pressure field
    # U1_high = np.stack(U1_high_list)  # high-fidelity x-velocity field
    # U2_high = np.stack(U2_high_list)  # high-fidelity y-velocity field
    high_data = {
        "field": P_high[: args.n_samples].reshape(args.n_samples, -1),
        "condition": K_high[: args.n_samples].reshape(args.n_samples, -1),
        "domain": domain,
        "resolution": args.m_high,
    }
    print("Generated high-fidelity solutions...")

    # Solve the Darcy flow problem for low-fidelity
    A_reg_low = enforce_integral(args.m_high // args.ss_factor)
    results_low = Parallel(n_jobs=args.threads)(
        delayed(generate_darcy_solution)(
            K_low[i], A_reg_low, args.m_high // args.ss_factor, i
        )
        for i in range(args.resume, total_samples)
    )
    P_low_list, U1_low_list, U2_low_list = zip(*results_low)
    P_low = np.stack(P_low_list)  # low-fidelity pressure field
    U1_low = np.stack(U1_low_list)  # low-fidelity x-velocity field
    U2_low = np.stack(U2_low_list)  # low-fidelity y-velocity field
    print("Generated low-fidelity solutions...")

    # Query the low-fieldity solution at the high-fidelity domain
    results_recon = Parallel(n_jobs=args.threads)(
        delayed(query_lf_at_hf_domain)(
            K_low[i], P_low[i], U1_low[i], U2_low[i], args.m_high, args.ss_factor
        )
        for i in range(args.resume, total_samples)
    )
    K_recon_list, P_recon_list, U1_recon_list, U2_recon_list = zip(*results_recon)
    K_recon = np.stack(K_recon_list)  # recon-fidelity permeability field
    P_recon = np.stack(P_recon_list)  # recon-fidelity pressure field
    # U1_recon = np.stack(U1_recon_list)  # recon-fidelity x-velocity field
    # U2_recon = np.stack(U2_recon_list)  # recon-fidelity y-velocity field
    low_data = {
        "field": P_recon[: args.n_samples].reshape(args.n_samples, -1),
        "condition": K_recon[: args.n_samples].reshape(args.n_samples, -1),
        "domain": domain,
        "resolution": args.m_high,
    }
    print("Interpolated low-fidelity solutions to high-fidelity domain...")

    # Test data

    test_data = {
        "LF_field": P_recon[args.n_samples :].reshape(args.n_test_samples, -1),
        "HF_field": P_high[args.n_samples :].reshape(args.n_test_samples, -1),
        "LF_solved": P_low[args.n_samples :].reshape(args.n_test_samples, -1),
        "condition": K_high[args.n_samples :].reshape(args.n_test_samples, -1),
        "domain": domain,
        "resolution": args.m_high,
    }

    # plot a snapshot
    plot_snapshot(
        domain,
        K_high,
        K_low,
        K_recon,
        P_high,
        P_low,
        P_recon,
        args.m_high,
        args.ss_factor,
    )

    # save
    np.savez("high_fidelity.npz", **high_data)
    np.savez("low_fidelity.npz", **low_data)
    np.savez("test_data.npz", **test_data)


if __name__ == "__main__":
    args = parse_args()
    generate(args)
