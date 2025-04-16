import numpy as np
import argparse
from argparse import Namespace
from sklearn import gaussian_process as gp
from scipy import interpolate

parser = argparse.ArgumentParser(
    prog="Poisson", description="Poisson equaiton data generators"
)
parser.add_argument(
    "-n", "--n_samples", type=int, default=2000, help="Number of samples to generate"
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
print(
    f"Generating {args.n_samples} samples for High fidelity (m: {args.m_high})"
    "and Low fideltiy (m: {args.m_low})"
)


# Solve the Poisson Equation for Generating Data
def solver(f, N):
    """Solve for the Poisson equation
    Inputs:
        f: Forcing term
        N: Number of discretization points (including boundary)
    """
    h = 1 / (N - 1)
    K = -2 * np.eye(N - 2) + np.eye(N - 2, k=1) + np.eye(N - 2, k=-1)
    b = h**2 * 20 * f[1:-1]
    u = np.linalg.solve(K, b)
    u = np.concatenate(([0], u, [0]))
    return u


# GP
class GRF:
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


# DataGeneration
def generate_data(args: Namespace):
    N = args.m_high * 10
    space = GRF(1, length_scale=0.05, N=N, interp="cubic")
    features = space.random(args.n_samples)
    domain_high = np.linspace(0, 1, args.m_high)
    domain_low = np.linspace(0, 1, args.m_low)
    features_high = space.eval_u(features, domain_high)
    features_low = space.eval_u(features, domain_low)

    # High Fidelity Data
    x_high, y_high, y_high_domain = [], [], []
    for ii in range(args.n_samples):
        sol = solver(features_high[ii], args.m_high)
        idx = np.random.choice(args.m_high, 1, replace=False)
        x_high.append(domain_high[idx].item())
        y_high.append(sol[idx].item())
        y_high_domain.append(sol)

    x_high = np.array(x_high)  # High fidelity random sensor
    y_high = np.array(y_high)  # High fidelity solution at random sensor
    y_high_domain = np.array(y_high_domain)  # High fidelity solution

    high_data = {}
    high_data["x_high"] = x_high  # Random sampled sensor
    high_data["y_high"] = y_high  # Field at randomly sampled sensor
    high_data["y_high_at_domain"] = y_high_domain  # Full high field
    high_data["features"] = features_high  # Features
    high_data["domain"] = domain_high  # Full domain
    high_data["n_sensors"] = args.m_high  # Number of sensors

    # Low Fidelity Data
    y_low, y_low_at_x_high, y_low_at_domain_high = [], [], []
    for ii in range(args.n_samples):
        sol = solver(features_low[ii], args.m_low)
        interp = interpolate.interp1d(
            domain_low, sol, kind="cubic", copy=False, assume_sorted=True
        )
        y_low_at_x_high.append(interp(x_high[ii]))
        y_low_at_domain_high.append(interp(domain_high))
        y_low.append(sol)

    y_low_at_x_high = np.array(
        y_low_at_x_high
    )  # Low fidelity solution at high random sensor
    y_low_at_domain_high = np.array(
        y_low_at_domain_high
    )  # Low fidelity solution at high domain
    y_low = np.array(y_low)  # Low fideltiy solution

    low_data = {}
    # Low fidelity field at high fidelity sensor
    low_data["y_low_at_x_high"] = y_low_at_x_high
    # Low fideltiy field at high domain
    low_data["y_low_at_domain_high"] = y_low_at_domain_high
    # Low fideltiy field at low domain
    low_data["y_low"] = y_low
    # Features sub-sampled on low domain
    low_data["features"] = features_low
    # Low domain
    low_data["domain"] = domain_low
    low_data["n_sensors"] = args.m_low  # Number of sensors

    # Save data
    np.savez("low_fideltiy.npz", **low_data)
    np.savez("high_fideltiy.npz", **high_data)
    print("Data saved to low_fidelity.npz and high_fidelity.npz")


if __name__ == "__main__":
    # generate data
    generate_data(args)
