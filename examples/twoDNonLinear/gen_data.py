# examples/twoDNonLinear/gen_data.py
"""
Implementation of the 2D model with non-linear correlation as described in the paper.
Modified from Thakur, A., Tripura, T. and Chakraborty, S., 2022.
Multi-fidelity wavelet neural operator with application to uncertainty quantification.
arXiv preprint arXiv:2208.05606.
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# from scipy import interpolate
plt.style.use("../../scripts/journal.mplstyle")

parser = argparse.ArgumentParser(
    prog="TwoDCorrelation",
    description="generate synthetic data for 1d model with correlation with the input",
)
parser.add_argument(
    "-n", "--n_samples", type=int, default=3000, help="Number of samples to generate"
)

parser.add_argument(
    "-nt", "--n_test_samples", type=int, default=200, help="Number of samples to test"
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
    """Evaluate the high fidelity model at a given point x with parameters a.
    Args:
        a (np.ndarray): Input feature, shape (n_samples, n_eval_points).
        domain (np.ndarray): Domain points, shape (n_eval_points, 2).
    Returns:
        np.ndarray: High fidelity solution, shape (n_samples, n_eval_points).
    """
    x, y = domain[:, 0], domain[:, 1]
    # original
    # hf_solution = np.cos(a) * np.cos(domain[:, 1]) ** 2

    # modified
    hf_solution = (
        (1 + 0.5 * np.sin(3 * x) + 0.3 * np.sin(6 * y)) * np.cos(a) * np.cos(y) ** 2
        + 0.1 * np.sin(25 * (x + y))  # diagonal high-freq oscillation
        + 0.05
        * np.exp(-150 * ((x - 0.75) ** 2 + (y - 0.25) ** 2))  # sharp Gaussian bump
        + 0.03 * np.heaviside(np.sin(5 * y - 2 * x), 1.0)  # discontinuous stripe
        + 0.02 * np.sign(np.sin(3 * x) * np.cos(2 * y))  # piecewise smooth oscillation
    )

    assert hf_solution.shape == a.shape
    return hf_solution


def eval_low_fidelity_model(a: np.ndarray, domain: np.ndarray):
    """Evaluate the low fidelity model at a given point x with parameters a.
    Args:
        a (np.ndarray): Input feature, shape (n_samples, n_eval_points).
        domain (np.ndarray): Domain points, shape (n_eval_points, 2).
    Returns:
        np.ndarray: Low fidelity solution, shape (n_samples, n_eval_points).
    """
    x, y = domain[:, 0], domain[:, 1]
    # original
    # lf_solution = np.cos(a) * np.cos(domain[:, 1]) + domain[:, 0]

    # modified
    lf_solution = (
        np.cos(a) * np.cos(y)
        + 0.5 * x
        + 0.05 * np.sin(5 * x + 2 * y) * np.cos(3 * y)
        - 0.02
        * np.exp(
            -30 * ((x - 0.25) ** 2 + (y - 0.75) ** 2)
        )  # bump in different location
    )

    assert lf_solution.shape == a.shape
    return lf_solution


def sample_input_function(n_samples: int, domain: np.ndarray):
    """sample the input function"""
    k = np.random.uniform(args.k_range[0], args.k_range[1], n_samples).reshape(-1, 1)
    x, y = domain[:, 0], domain[:, 1]
    # original
    # a = k * domain[:, 0] ** 2 - 4.0
    # modfied
    a = (
        k
        * (
            x**2
            + 0.5 * np.sin(3 * x**2) * np.cos(2 * y**3)
            + 0.2 * np.sin(10 * x * y)
        )
        - 4.0
    )
    return a


def plot_snapshot(
    domain: np.ndarray,
    features: np.ndarray,
    hf_solution: np.ndarray,
    lf_solution: np.ndarray,
):
    """Plot a snapshot of the high and low fidelity solutions."""
    idx_plot = 0  # for reproducibility, always plot the first sample

    # reshape for plotting
    features = features.reshape(-1, args.m_high, args.m_high)
    hf_solution = hf_solution.reshape(-1, args.m_high, args.m_high)
    lf_solution = lf_solution.reshape(-1, args.m_high, args.m_high)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    vmin = min(hf_solution.min(), lf_solution.min())
    vmax = max(hf_solution.max(), lf_solution.max())
    axs[0].imshow(
        hf_solution[idx_plot],
        extent=(
            domain[:, 0].min(),
            domain[:, 0].max(),
            domain[:, 1].min(),
            domain[:, 1].max(),
        ),
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axs[0].set_title("High-fidelity")

    a2 = axs[1].imshow(
        lf_solution[idx_plot],
        extent=(
            domain[:, 0].min(),
            domain[:, 0].max(),
            domain[:, 1].min(),
            domain[:, 1].max(),
        ),
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(a2, ax=axs[1], orientation="vertical", fraction=0.046, pad=0.04)
    axs[1].set_title("Low-fidelity")

    for ax in axs:
        ax.label_outer()
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
    plt.savefig("snapshot.png", dpi=300, bbox_inches="tight")


def plot_joint(lf_solution, hf_solution):
    """Plot joint distribution of high and low fidelity solutions"""
    sns.jointplot(
        x=lf_solution.flatten(),
        y=hf_solution.flatten(),
        kind="hex",
        color="blue",
        marginal_kws=dict(bins=50, fill=True),
    )
    plt.xlabel("Low-fidelity Solution")
    plt.ylabel("High-fidelity Solution")
    plt.tight_layout()
    plt.savefig("joint_distribution.png", dpi=300, bbox_inches="tight")


def generate():
    """generate the data"""
    x = np.linspace(0, 1, args.m_high)
    y = np.linspace(0, 1, args.m_high)
    XX, YY = np.meshgrid(x, y)
    domain = np.stack([XX.flatten(), YY.flatten()], axis=1)  # shape (m_high^2, 2)
    total_samples = args.n_samples + args.n_test_samples
    features = sample_input_function(
        total_samples, domain
    )  # samples of the input functions
    hf_solution = eval_high_fidelity_model(
        features, domain
    )  # evaluate high fidelity model
    lf_solution = eval_low_fidelity_model(
        features, domain
    )  # evaluate low fidelity model

    # plot joint distribution
    plot_joint(lf_solution, hf_solution)

    high_data = {
        "field": hf_solution[: args.n_samples],  # only high fidelity for training,
        "condition": features[: args.n_samples],  # input function for training
        "domain": domain,
        "resolution": args.m_high,
    }

    low_data = {
        "field": lf_solution[: args.n_samples],  # only low fidelity for training
        "condition": features[: args.n_samples],  # input function for training
        "domain": domain,
        "resolution": args.m_high,
    }

    test_data = {
        "LF_field": lf_solution[args.n_samples :],
        "HF_field": hf_solution[args.n_samples :],
        "condition": features[args.n_samples :],
        "domain": domain,
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
