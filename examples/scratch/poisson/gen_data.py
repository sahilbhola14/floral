# examples/poisson/gen_data.py
"""
1D Poisson equation example

Reference:
Lu, L., Pestourie, R., Johnson, S.G. and Romano, G., 2022.
Multifidelity deep neural operators for efficient learning of partial differential
equations with application to fast inverse design of nanoscale heat transport.
Physical Review Research, 4(2), p.023210.
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn import gaussian_process as gp
from scipy import interpolate
from tqdm import tqdm

plt.style.use("../../scripts/journal.mplstyle")


class GRF:
    """Gaussian Random Field class for generating samples of a random field"""

    def __init__(self, T, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.T = T
        self.kernel = kernel
        self.length_scale = length_scale
        self.N = N
        self.x = np.linspace(0, T, self.N)[:, None]
        self.interp = interp
        kernel = gp.kernels.RBF(length_scale=self.length_scale)
        self.K = kernel(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n):
        """generate a random field sample"""
        return np.dot(self.L, np.random.randn(self.N, n)).T

    def eval_u(self, ys, sensors):
        """evaluate the random field at the sensor locations"""
        y_interp = []
        for y in ys:
            interp = interpolate.interp1d(
                self.x.ravel(), y, kind=self.interp, copy=False, assume_sorted=True
            )
            y_interp.append(interp(sensors))

        return np.array(y_interp)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="Poisson", description="Generate data for 1D Poisson equation example"
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=2000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "-ml",
        "--m_low",
        type=int,
        default=10,
        help="Number of discretization points for low fidelity",
    )
    parser.add_argument(
        "-mh",
        "--m_high",
        type=int,
        default=100,
        help="Number of discretization points for high fidelity",
    )

    args = parser.parse_args()
    print("==" * 20)
    print("Poisson 1D example")
    print(f"Number of training samples: {args.n_samples}")
    print(f"Low fidelity discretization points: {args.m_low}")
    print(f"High fidelity discretization points: {args.m_high}")
    print("==" * 20)

    return args


def solver(f, N):
    """Solve the Poisson equation with a given source term f.
    Args:
        f (np.ndarray): Source term.
        N (int): Number of discretization points.
    Returns:
        np.ndarray: Solution to the Poisson equation.
    """
    h = 1 / (N - 1)
    K = -2 * np.eye(N - 2) + np.eye(N - 2, k=1) + np.eye(N - 2, k=-1)
    b = h**2 * 20 * f[1:-1]
    u = np.linalg.solve(K, b)
    u = np.concatenate(([0], u, [0]))
    return u


def plot_snapshot(
    domain: np.ndarray,
    features: np.ndarray,
    hf_solution: np.ndarray,
    lf_solution: np.ndarray,
    random: bool = False,
):
    """Plot a snapshot of the high and low fidelity solutions."""
    if random:
        idx_plot = np.random.randint(0, hf_solution.shape[0])
    else:
        idx_plot = 0  # for reproducibility, always plot the first sample
    print(f"Plotting sample index: {idx_plot}")
    plt.figure(figsize=(6, 2.5))
    plt.plot(
        domain.ravel(),
        lf_solution[idx_plot, :],
        label="Low-fidelity",
        color="grey",
        alpha=0.6,
    )
    plt.plot(domain.ravel(), hf_solution[idx_plot, :], label="High-fidelity", color="k")
    plt.xlabel(r"$x_w$")
    plt.ylabel(r"$w(x_w)$")
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig("data_snapshot.png")


def generate(args):
    """Generate data for the Poisson equation example.
    Notes:
        - For low fidelity, hf_features is used when saving because the lf_features
        are obtained by interpolation from the high fidelity features. We basicallly
        need the features interpolated to the high fidelity domain for training.
    """
    N = args.m_high * 10
    total_samples = args.n_samples
    space = GRF(1, length_scale=0.05, N=N, interp="cubic")
    features = space.random(total_samples)  # Generate random features
    hf_domain = np.linspace(0, 1, args.m_high)  # High fidelity domain
    lf_domain = np.linspace(0, 1, args.m_low)  # Low fidelity domain
    hf_features = space.eval_u(features, hf_domain)  # High fidelity features
    lf_features = space.eval_u(features, lf_domain)  # Low fidelity features

    # High Fidelity Data
    hf_solution = []
    pbar = tqdm(range(total_samples), desc="High fidelity")
    for ii in pbar:
        sol = solver(hf_features[ii], args.m_high)
        hf_solution.append(sol)
    hf_solution = np.array(hf_solution)
    pbar.close()

    # Low Fidelity Data
    lf_solution = []
    pbar = tqdm(range(total_samples), desc="Low fidelity")
    for ii in pbar:
        sol = solver(lf_features[ii], args.m_low)
        interp = interpolate.interp1d(
            lf_domain, sol, kind="cubic", copy=False, assume_sorted=True
        )
        lf_solution.append(interp(hf_domain))

    lf_solution = np.array(lf_solution)
    pbar.close()
    # plot a snapshot
    plot_snapshot(hf_domain, hf_features, hf_solution, lf_solution)
    # reshape and save data
    hf_domain = hf_domain.reshape(-1, 1)
    hf_features = np.expand_dims(hf_features, 1)  # (B, channels_c, *dim_c)
    hf_solution = np.expand_dims(hf_solution, 1)  # (B, channels_f, *dim_f)
    lf_solution = np.expand_dims(lf_solution, 1)  # (B, channels_f, *dim_f)

    assert hf_solution.shape == lf_solution.shape, (
        "Incorrect lf solution shape."
        "Consider interpolating the low fidelity solution."
    )

    high_data = {
        "field": hf_solution,
        "condition": hf_features,
        "field_domain": hf_domain,
        "condition_domain": hf_domain,
    }

    low_data = {
        "field": lf_solution,
        "condition": hf_features,
        "field_domain": hf_domain,
        "condition_domain": hf_domain,
    }

    # save
    np.savez("high_fidelity.npz", **high_data)
    np.savez("low_fidelity.npz", **low_data)

    print("saved high fidelity data to high_fidelity.npz")
    print("saved low fidelity data to low_fidelity.npz")


if __name__ == "__main__":
    args = parse_args()  # Parse command line arguments
    generate(args)  # Generate data for the Poisson equation example
