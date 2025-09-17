# /examples/oneDCorr/gen_data.py
"""
Generate synthetic data for a 1D model with correlation between the
low and high fidelity models.
Reference: Thakur, A., Tripura, T. and Chakraborty, S., 2022. Multi-fidelity wavelet
neural operator with application to uncertainty quantification.
arXiv preprint arXiv:2208.05606.
"""
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

print(f"Generating {args.n_samples} samples for training and validation")


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
    features_plot = features[idx_plot, :]
    lf_solution_plot = lf_solution[idx_plot, :]
    hf_solution_plot = hf_solution[idx_plot, :]

    fig, axs = plt.subplots(1, 2, figsize=(6, 2.5), sharex=True)
    axs[0].plot(domain, features_plot, color="k")
    axs[0].set_ylabel(r"$a(x)$")
    axs[1].plot(domain, hf_solution_plot, color="k", label="High-fidelity")
    axs[1].plot(domain, lf_solution_plot, color="grey", alpha=0.6, label="Low-fidelity")
    axs[1].legend(fontsize=10, loc="upper right")
    axs[1].set_ylabel(r"$w(x)$")
    for ax in axs:
        ax.set_xlabel("$x$")

    plt.legend(fontsize=10, loc="upper right")
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
        "field": hf_solution[: args.n_samples],
        "field_domain": domain.reshape(-1, 1),
        "condition": features[: args.n_samples],
        "condition_domain": domain.reshape(-1, 1),  # domain for the input function
        "resolution": args.m_high,
    }

    low_data = {
        "field": lf_solution[: args.n_samples],
        "field_domain": domain.reshape(-1, 1),
        "condition": features[: args.n_samples],
        "condition_domain": domain.reshape(-1, 1),  # domain for the input function
        "resolution": args.m_high,
    }

    # plot a snapshot
    plot_snapshot(domain, features, hf_solution, lf_solution)

    # save
    np.savez("high_fidelity.npz", **high_data)
    np.savez("low_fidelity.npz", **low_data)


if __name__ == "__main__":
    generate()
