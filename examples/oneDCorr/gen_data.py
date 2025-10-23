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
    "-n", "--n_samples", type=int, default=10000, help="Number of samples to generate"
)

parser.add_argument(
    "-mh",
    "--m_high",
    type=int,
    default=128,
    help="Number of discretization points for high fidelity",
)
parser.add_argument(
    "--k_range", type=list, default=[10, 14], help="Range of k values to sample from"
)
parser.add_argument("--plot", action="store_true", help="Plot a snapshot of the data")

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
    condition: np.ndarray,
    hf_solution: np.ndarray,
    lf_solution: np.ndarray,
):
    """Plot a snapshot of the high and low fidelity solutions."""
    idx_plot = 0  # for reproducibility, always plot the first sample
    condition_plot = condition[idx_plot, :].ravel()
    lf_solution_plot = lf_solution[idx_plot, :].ravel()
    hf_solution_plot = hf_solution[idx_plot, :].ravel()
    domain_plot = domain.ravel()

    fig, axs = plt.subplots(1, 2, figsize=(6, 2.5), sharex=True)
    axs[0].plot(domain_plot, condition_plot, color="k")
    axs[0].set_ylabel(r"$a(x)$")
    axs[0].set_ylim(-10, 10)
    axs[1].plot(domain_plot, hf_solution_plot, color="k", label="High-fidelity")
    axs[1].plot(
        domain_plot, lf_solution_plot, color="grey", alpha=0.6, label="Low-fidelity"
    )
    axs[1].legend(fontsize=10)
    axs[1].set_ylabel(r"$w(x)$")
    axs[1].set_ylim(bottom=-2, top=2)
    for ax in axs:
        ax.set_xlabel("$x$")

    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig("data_snapshot.png")


def generate():
    """generate the data"""
    domain = np.linspace(0, 1, args.m_high)  # high fidelity domain
    condition = sample_input_function(
        args.n_samples, domain
    )  # samples of the input functions
    hf_solution = eval_high_fidelity_model(
        condition, domain
    )  # evaluate high fidelity model
    lf_solution = eval_low_fidelity_model(
        condition, domain
    )  # evaluate low fidelity model

    # reshape
    domain = domain.reshape(-1, 1)  # flattened domain
    condition = np.expand_dims(condition, 1)  # (B, channel_c, *dim_c)
    hf_solution = np.expand_dims(hf_solution, 1)  # (B, channels_f, *dim_f)
    lf_solution = np.expand_dims(lf_solution, 1)  # (B, channels_f, *dim_f)

    assert hf_solution.shape == lf_solution.shape, (
        "Incorrect lf solution shape."
        "Consider interpolating the low fidelity solution."
    )

    high_data = {
        "field": hf_solution[: args.n_samples],
        "condition": condition[: args.n_samples],
        "field_domain": domain,
        "condition_domain": domain,
    }
    # low fidelity must be interpoalted to the same domain as HF
    low_data = {
        "field": lf_solution[: args.n_samples],
        "condition": condition[: args.n_samples],
        "field_domain": domain,
        "condition_domain": domain,
    }

    # plot a snapshot
    if args.plot:
        plot_snapshot(domain, condition, hf_solution, lf_solution)

    # save
    np.savez("high_fidelity.npz", **high_data)
    np.savez("low_fidelity.npz", **low_data)

    print("saved high fidelity data to high_fidelity.npz")
    print("saved low fidelity data to low_fidelity.npz")


if __name__ == "__main__":
    generate()
