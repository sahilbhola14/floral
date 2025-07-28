# scripts/twoDNonLinear/plot.py
"""
Plot HF mean removed 2D fields.
TODO:
    - Remove HF mean from the plots.
"""
import torch
import math
import matplotlib.pyplot as plt

plt.style.use("../journal.mplstyle")

# Begin user input
n_samples = 200
n_sensors = 100
plot_idx = {"automatic": True, "idx": 6}
plot_error = False  # Whether to plot error or not
# End user input


def find_best_idx(true, prediction):
    """
    Find the index with smallest error between true and predicted values.
    """
    assert (
        true.shape == prediction.shape
    ), "True and prediction must have the same shape."
    error = (true - prediction).abs().mean(dim=1)
    idx = error.argmin().item()
    print(f"Best index found: {idx} with error {error.min().item()}")
    return idx


def find_worst_idx(true, prediction):
    """Find the index with largest error between true and predicted values."""
    assert (
        true.shape == prediction.shape
    ), "True and prediction must have the same shape."
    error = (true - prediction).abs().mean(dim=1)
    idx = error.argmax().item()
    print(f"Worst index found: {idx} with error {error.max().item()}")
    return idx


print(f"Plotting fields for {n_samples} samples and {n_sensors} sensors...")

# Load data
data = torch.load(f"twoDNonLinear_{n_samples}_samples_{n_sensors}_sensors_results.pt")
data_mfFlow = torch.load(
    f"twoDNonLinear_{n_samples}_samples_{n_sensors}_sensors_results_mfFlow.pt"
)
data_gp = torch.load(
    f"twoDNonLinear_{n_samples}_samples_{n_sensors}_sensors_GP_results.pt"
)
data_gp_mfFlow = torch.load(
    f"twoDNonLinear_{n_samples}_samples_{n_sensors}_sensors_GP_results_mfFlow.pt"
)

field = data["field"]
field_mfFlow = data_mfFlow["field"]
field_gp = data_gp["field"]
field_gp_mfFlow = data_gp_mfFlow["field"]

domain = data_mfFlow["domain"].ravel()

plot_idx = (
    plot_idx["idx"]
    if not plot_idx["automatic"]
    else find_best_idx(true=field["HF_field"], prediction=field["Prediction"].mean(1))
)
# plot_idx = (
#     plot_idx["idx"]
#     if not plot_idx["automatic"]
#     else find_worst_idx(
#         true=field["HF_field"], prediction=field_gp["Prediction"]["mean"]
#     )
# )

# Extract relevant data
LF_field = field_gp.get("LF_field")[plot_idx]
HF_field = field_gp.get("HF_field")[plot_idx]

Prediction_flora = field.get("Prediction")[plot_idx]
mean_flora = Prediction_flora.mean(dim=0)
std_flora = Prediction_flora.std(dim=0)

Prediction_floren = field_mfFlow.get("Prediction")[plot_idx]
mean_floren = Prediction_floren.mean(dim=0)
std_floren = Prediction_floren.std(dim=0)

Prediction_gp = field_gp.get("Prediction")
mean_gp = Prediction_gp["mean"][plot_idx]
std_gp = Prediction_gp["std"][plot_idx]

Prediction_regp = field_gp_mfFlow.get("Prediction")
mean_regp = Prediction_regp["mean"][plot_idx]
std_regp = Prediction_regp["std"][plot_idx]

image_dim = int(math.sqrt(mean_regp.shape[-1]))
assert image_dim * image_dim == mean_regp.shape[-1]


# Ordered list of methods
method_order = ["FLORA", "FLOREN", "GP", "REGP"]

# container fo results
means = {
    "FLORA": mean_flora,
    "FLOREN": mean_floren,
    "GP": mean_gp,
    "REGP": mean_regp,
}

stds = {
    "FLORA": std_flora,
    "FLOREN": std_floren,
    "GP": std_gp,
    "REGP": std_regp,
}

# plots
# range of values for mean and std
mean_vmin, mean_vmax = float("inf"), float("-inf")
std_vmin, std_vmax = float("inf"), float("-inf")
for ii, method in enumerate(method_order):
    if plot_error:
        mean_vmin = min(mean_vmin, (HF_field - means[method]).min())
        mean_vmax = max(mean_vmax, (HF_field - means[method]).max())
        mean_vmin, mean_vmax = (
            -0.1,
            0.1,
        )  # manually set for error plots to visualization
        std_vmin = min(std_vmin, stds[method].log10().min())
        std_vmax = max(std_vmax, stds[method].log10().max())
    else:
        mean_vmin = min(mean_vmin, means[method].min())
        mean_vmax = max(mean_vmax, means[method].max())
        std_vmin = min(std_vmin, stds[method].log10().min())
        std_vmax = max(std_vmax, stds[method].log10().max())

fig, axs = plt.subplots(
    1, 2, figsize=(10, 5), dpi=300, constrained_layout=True, sharex=True, sharey=True
)
axs[0].imshow(
    HF_field.view(image_dim, image_dim),
    origin="lower",
    vmin=mean_vmin,
    vmax=mean_vmax,
    interpolation="bilinear",
)
axs[0].set_title("High-fidelity")
axs[1].imshow(
    LF_field.view(image_dim, image_dim),
    origin="lower",
    vmin=mean_vmin,
    vmax=mean_vmax,
    interpolation="bilinear",
)
axs[1].set_title("Low-fidelity")
plt.savefig("twoDNonLinear_HF_LF_fields.png")

fig, axs = plt.subplots(
    2, 4, figsize=(12, 5), sharey=True, sharex=True, constrained_layout=True, dpi=300
)

for ii, method in enumerate(method_order):
    # plot mean
    plot_mean = (
        (HF_field - means[method]).view(image_dim, image_dim)
        if plot_error
        else means[method].view(image_dim, image_dim)
    )
    ax_mean = axs[0, ii].imshow(
        plot_mean,
        extent=(0, 1, 0, 1),
        origin="lower",
        aspect="equal",
        vmin=mean_vmin,
        vmax=mean_vmax,
        interpolation="bilinear",
    )
    # plot std
    plot_std = stds[method].view(image_dim, image_dim).log10()
    ax_std = axs[1, ii].imshow(
        plot_std,
        extent=(0, 1, 0, 1),
        origin="lower",
        aspect="equal",
        vmin=std_vmin,
        vmax=std_vmax,
        interpolation="bilinear",
    )
    # set titles
    axs[0, ii].set_title(method)

# Add colorbar for row 0 (mean)
cbar_ax_mean = fig.add_axes([1, 0.56, 0.015, 0.34])  # [left, bottom, width, height]
cbar_mean = fig.colorbar(ax_mean, cax=cbar_ax_mean)
cbar_mean.set_label(r"$w(a) - \hat{w}(a)$") if plot_error else cbar_mean.set_label(
    r"$\hat{w}(a)$"
)

# Add colorbar for row 1 (std)
cbar_ax_std = fig.add_axes([1, 0.11, 0.015, 0.34])  # adjust to match lower row
cbar_std = fig.colorbar(ax_std, cax=cbar_ax_std)
cbar_std.set_label(r"$\log_{10}(\sigma)$")

for ii, ax in enumerate(axs.flatten()):
    # plot outer labels
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.label_outer()

plt.savefig("twoDNonLinear_comparison.png")
plt.close()
