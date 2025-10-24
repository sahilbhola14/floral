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
from scipy.stats import pearsonr

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
    "-nt", "--n_test_samples", type=int, default=100, help="Number of samples to test"
)
parser.add_argument(
    "-mh",
    "--m_high",
    type=int,
    default=50,
    help="Number of discretization points for high fidelity",
)
parser.add_argument(
    "--k_range", type=list, default=[-2, 2], help="Range of k values to sample from"
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
    y = domain[:, 1]
    # original
    hf_solution = np.cos(a) * np.cos(y) ** 2

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
    lf_solution = np.cos(a) * np.cos(y) + x

    assert lf_solution.shape == a.shape
    return lf_solution


def sample_input_function(n_samples: int, domain: np.ndarray):
    """sample the input function"""
    k = np.random.uniform(args.k_range[0], args.k_range[1], n_samples).reshape(-1, 1)
    x = domain[:, 0]
    # original
    a = k * x - 4.0

    idx_plot = np.random.choice(len(a), 10, replace=False)
    fig, ax = plt.subplots(
        2, 5, figsize=(10, 4), layout="constrained", sharex=True, sharey=True, dpi=300
    )
    for i, ax_ in enumerate(ax.flat):
        ax_.imshow(
            a[idx_plot[i]].reshape(args.m_high, args.m_high),
            extent=(
                domain[:, 0].min(),
                domain[:, 0].max(),
                domain[:, 1].min(),
                domain[:, 1].max(),
            ),
            origin="lower",
            aspect="equal",
            interpolation="bilinear",
        )
        ax_.set_xlabel(r"$x_a$")
        ax_.set_ylabel(r"$y_a$")
        ax_.label_outer()
        ax_.grid(False, which="both")
        ax_.xaxis.grid(False, which="both")
        ax_.yaxis.grid(False, which="both")
    plt.savefig("input_function_samples.png", dpi=300)

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

    # plot field
    features_plot = features[idx_plot]
    hf_solution_plot = hf_solution[idx_plot]
    lf_solution_plot = lf_solution[idx_plot]

    vmin = min(hf_solution_plot.min(), lf_solution_plot.min())
    vmax = max(hf_solution_plot.max(), lf_solution_plot.max())

    fig, axs = plt.subplots(
        1, 3, figsize=(12, 5), layout="constrained", sharex=True, sharey=True
    )

    # --- Plot features with horizontal colorbar ---
    im0 = axs[0].imshow(
        features_plot,
        extent=(
            domain[:, 0].min(),
            domain[:, 0].max(),
            domain[:, 1].min(),
            domain[:, 1].max(),
        ),
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
    )
    axs[0].set_title("Input")
    cb0 = fig.colorbar(im0, ax=axs[0], orientation="vertical", pad=0.01, shrink=0.5)
    cb0.set_label(r"$a$")

    # --- Plot high-fidelity solution ---
    axs[1].imshow(
        hf_solution_plot,
        extent=(
            domain[:, 0].min(),
            domain[:, 0].max(),
            domain[:, 1].min(),
            domain[:, 1].max(),
        ),
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    axs[1].set_title("High-fidelity")

    # --- Plot low-fidelity solution ---
    im2 = axs[2].imshow(
        lf_solution_plot,
        extent=(
            domain[:, 0].min(),
            domain[:, 0].max(),
            domain[:, 1].min(),
            domain[:, 1].max(),
        ),
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    axs[2].set_title("Low-fidelity")

    # --- Shared vertical colorbar for hf and lf ---
    cb1 = fig.colorbar(im2, ax=axs[1:], orientation="vertical", shrink=0.5, pad=0.01)
    cb1.set_label(r"$w$")

    for ia, ax in enumerate(axs):
        if ia == 0:
            ax.set_xlabel(r"$x_a$")
            ax.set_ylabel(r"$y_a$")
        else:
            ax.set_xlabel(r"$x_w$")
            ax.set_ylabel(r"$y_w$")
        ax.label_outer()
        ax.grid(False, which="both")
        ax.xaxis.grid(False, which="both")
        ax.yaxis.grid(False, which="both")
    plt.savefig("snapshot.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_joint(lf_solution, hf_solution):
    """Plot joint distribution of high and low fidelity solutions"""

    # Flatten data
    x = lf_solution.flatten()
    y = hf_solution.flatten()

    # Compute Pearson correlation
    r, _ = pearsonr(x, y)

    # Create the plot
    g = sns.jointplot(
        x=x,
        y=y,
        kind="hex",  # You can also use "scatter" or "reg"
        color="blue",
        marginal_kws=dict(bins=50, fill=True),
    )

    # Add y = x line (diagonal)
    g.ax_joint.plot([x.min(), x.max()], [x.min(), x.max()], "r--", lw=2)

    # Annotate correlation
    g.ax_joint.text(
        0.05,
        0.95,
        f"$r = {r: .2f}$",
        transform=g.ax_joint.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    # Labels and layout
    g.set_axis_labels("Low-fidelity Solution", "High-fidelity Solution")
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

    # check nan
    assert not np.isnan(hf_solution).any(), "High fidelity solution contains NaN"
    assert not np.isnan(lf_solution).any(), "Low fidelity solution contains NaN"

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
