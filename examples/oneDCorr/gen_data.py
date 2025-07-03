import numpy as np
import argparse
import matplotlib.pyplot as plt

plt.style.use("../../scripts/journal.mplstyle")

parser = argparse.ArgumentParser(
    prog="OneDCorrelation",
    description="generate synthetic data for 1d model with correlation with the input",
)
parser.add_argument(
    "-n", "--n_samples", type=int, default=2000, help="Number of samples to generate"
)

parser.add_argument(
    "-nt",
    "--n_test_samples",
    type=int,
    default=50,
    help="Number of samples to generate test",
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

print(
    f"Generating {args.n_samples} samples for training and validation, "
    f"and {args.n_test_samples} for testing."
)


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
    idx_plot = 0  # for reproducibility, always plot the first sample
    plt.figure(figsize=(8, 4))
    plt.plot(domain, lf_solution[idx_plot, :], label="Low-fidelity", color="blue")
    plt.plot(domain, hf_solution[idx_plot, :], label="High-fidelity", color="k")
    plt.xlabel("$x$")
    plt.ylabel("$w(a)$")
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig("snapshot.png")


def generate():
    """generate the data"""
    domain = np.linspace(0, 1, args.m_high)  # high fidelity domain
    features = sample_input_function(
        args.n_samples + args.n_test_samples, domain
    )  # samples of the input functions
    hf_solution = eval_high_fidelity_model(
        features, domain
    )  # evaluate high fidelity model
    lf_solution = eval_low_fidelity_model(
        features, domain
    )  # evaluate low fidelity model

    high_data = {
        "field": hf_solution[: args.n_samples],
        "condition": features[: args.n_samples],
        "domain": domain.reshape(-1, 1),
        "resolution": args.m_high,
    }

    low_data = {
        "field": lf_solution[: args.n_samples],
        "condition": features[: args.n_samples],
        "domain": domain.reshape(-1, 1),
        "resolution": args.m_high,
    }

    # Test config
    test_data = {
        "LF_field": lf_solution[args.n_samples :],
        "HF_field": hf_solution[args.n_samples :],
        "condition": features[args.n_samples :],
        "domain": domain.reshape(-1, 1),
        "resolution": args.m_high,
    }

    # plot a snapshot
    plot_snapshot(domain, features, hf_solution, lf_solution)

    # save
    np.savez("high_fidelity.npz", **high_data)
    np.savez("low_fidelity.npz", **low_data)
    np.savez("test_data.npz", **test_data)


if __name__ == "__main__":
    generate()
