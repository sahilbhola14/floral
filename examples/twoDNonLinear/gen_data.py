"""
Reference: Thakur, A., Tripura, T. and Chakraborty, S., 2022.
Multi-fidelity wavelet neural operator with application to uncertainty quantification.
arXiv preprint arXiv:2208.05606.
Implementation of the 2D model with non-linear correlation as described in the paper.
"""
import numpy as np
import argparse

# from scipy import interpolate

parser = argparse.ArgumentParser(
    prog="TwoDCorrelation",
    description="generate synthetic data for 1d model with correlation with the input",
)
parser.add_argument(
    "-n", "--n_samples", type=int, default=2000, help="Number of samples to generate"
)
parser.add_argument(
    "-mh",
    "--m_high",
    type=int,
    default=50,
    help="Number of discretization points for high fidelity",
)
parser.add_argument(
    "--k_range", type=list, default=[8, 10], help="Range of k values to sample from"
)
args = parser.parse_args()

print(f"Generating {args.n_samples} samples")


def eval_high_fidelity_model(a: np.ndarray, domain: np.ndarray):
    """Evaluate the high fidelity model at a given point x with parameters a."""
    return np.cos(a) * np.cos(domain[:, 1]) ** 2


def eval_low_fidelity_model(a: np.ndarray, domain: np.ndarray):
    """Evaluate the low fidelity model at a given point x with parameters a."""
    return np.cos(a) * np.cos(domain[:, 1]) + domain[:, 0]


def sample_input_function(n_samples: int, domain: np.ndarray):
    """sample the input function"""
    k = np.random.uniform(args.k_range[0], args.k_range[1], n_samples).reshape(-1, 1)
    return k * domain[:, 0] - 4.0


def generate():
    """generate the data"""
    x = np.linspace(0, 1, args.m_high)
    y = np.linspace(0, 1, args.m_high)
    XX, YY = np.meshgrid(x, y)
    domain = np.stack([XX.flatten(), YY.flatten()], axis=1)  # shape (m_high^2, 2)
    features = sample_input_function(
        args.n_samples, domain
    )  # samples of the input functions
    hf_solution = eval_high_fidelity_model(
        features, domain
    )  # evaluate high fidelity model
    lf_solution = eval_low_fidelity_model(
        features, domain
    )  # evaluate low fidelity model

    # reshape
    features = features.reshape(-1, args.m_high, args.m_high)
    hf_solution = hf_solution.reshape(-1, args.m_high, args.m_high)
    lf_solution = lf_solution.reshape(-1, args.m_high, args.m_high)

    high_data = {
        "domain": domain.reshape(1, -1),
        "features": features,
        "field": hf_solution,
    }

    low_data = {
        "domain": domain.reshape(1, -1),
        "features": features,
        "field": lf_solution,
    }

    # save
    np.savez("high_fidelity.npz", **high_data)
    np.savez("low_fidelity.npz", **low_data)


if __name__ == "__main__":
    generate()
