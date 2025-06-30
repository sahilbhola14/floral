import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.frameon": True,
        "legend.framealpha": 1,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "font.family": "serif",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
    }
)

parser = argparse.ArgumentParser(
    prog="OneDCorrelation",
    description="generate synthetic data for 1d model with correlation with the input",
)
parser.add_argument(
    "-n", "--n_samples", type=int, default=2000, help="Number of samples to generate"
)
parser.add_argument(
    "-mh",
    "--m_high",
    type=int,
    default=100,
    help="Number of discretization points for high fidelity",
)
parser.add_argument(
    "--k_range", type=list, default=[10, 14], help="Range of k values to sample from"
)
args = parser.parse_args()

print(f"Generating {args.n_samples} samples")


def eval_high_fidelity_model(a: np.ndarray, x: np.ndarray):
    """Evaluate the high fidelity model at a given point x with parameters a."""
    return np.sin(a)


def eval_low_fidelity_model(a: np.ndarray, x: np.ndarray):
    """Evaluate the low fidelity model at a given point x with parameters a."""
    return np.sin(a) + x.reshape(1, -1) - 0.25 * a


def sample_input_function(n_samples: int, domain: np.ndarray):
    """sample the input function"""
    k = np.random.uniform(args.k_range[0], args.k_range[1], n_samples).reshape(-1, 1)
    return k * domain - 4.0


def plot_snapshot(
    domain: np.ndarray,
    features: np.ndarray,
    hf_solution: np.ndarray,
    lf_solution: np.ndarray,
):
    """Plot a snapshot of the high and low fidelity solutions."""
    idx_plot = np.random.randint(0, features.shape[0])
    plt.figure()
    plt.plot(domain, hf_solution[idx_plot, :], label="High Fidelity")
    plt.plot(domain, lf_solution[idx_plot, :], label="Low Fidelity")
    plt.xlabel("x")
    plt.ylabel("w(a)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("snapshot.png")


def generate():
    """generate the data"""
    domain = np.linspace(0, 1, args.m_high)  # high fidelity domain
    features = sample_input_function(
        args.n_samples, domain
    )  # samples of the input functions
    hf_solution = eval_high_fidelity_model(
        features, domain
    )  # evaluate high fidelity model
    lf_solution = eval_low_fidelity_model(
        features, domain
    )  # evaluate low fidelity model

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

    # plot a snapshot
    plot_snapshot(domain, features, hf_solution, lf_solution)

    # save
    np.savez("high_fidelity.npz", **high_data)
    np.savez("low_fidelity.npz", **low_data)


if __name__ == "__main__":
    generate()
