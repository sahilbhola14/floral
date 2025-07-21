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
n_samples = 1000
n_sensors = 10
plot_idx = 6  # Select sample index
# End user input

# Load data
# data = torch.load(f"twoDNonLinear_{n_samples}_samples_{n_sensors}_sensors_results.pt")
# data_mfFlow = torch.load(
#     f"twoDNonLinear_{n_samples}_samples_{n_sensors}_sensors_results_mfFlow.pt"
# )
data_gp = torch.load(
    f"twoDNonLinear_{n_samples}_samples_{n_sensors}_sensors_GP_results.pt"
)
data_gp_mfFlow = torch.load(
    f"twoDNonLinear_{n_samples}_samples_{n_sensors}_sensors_GP_results_mfFlow.pt"
)

# field = data["field"]
# field_mfFlow = data_mfFlow["field"]
field_gp = data_gp["field"]
field_gp_mfFlow = data_gp_mfFlow["field"]

# domain = data_mfFlow["domain"].ravel()

# Extract relevant data
# LF_field = field_mfFlow.get("LF_field")[plot_idx]
# HF_field = field_mfFlow.get("HF_field")[plot_idx]

# Prediction_prono = field.get("Prediction")[plot_idx]
# mean_prono = Prediction_prono.mean(dim=0)
# std_prono = Prediction_prono.std(dim=0)

# Prediction_promino = field_mfFlow.get("Prediction")[plot_idx]
# mean_promino = Prediction_promino.mean(dim=0)
# std_promino = Prediction_promino.std(dim=0)

Prediction_gp = field_gp.get("Prediction")
mean_gp = Prediction_gp["mean"][plot_idx]
std_gp = Prediction_gp["std"][plot_idx]

Prediction_migp = field_gp_mfFlow.get("Prediction")
mean_migp = Prediction_migp["mean"][plot_idx]
std_migp = Prediction_migp["std"][plot_idx]

image_dim = int(math.sqrt(mean_migp.shape[-1]))
assert image_dim * image_dim == mean_migp.shape[-1]

mean_vmin = min(mean_gp.min(), mean_migp.min())
mean_vmax = max(mean_gp.max(), mean_migp.max())

std_vmin = min(std_gp.log10().min(), std_migp.log10().min())
std_vmax = max(std_gp.log10().max(), std_migp.log10().max())
# std_vmin, std_vmax = 0, 1

# Set up 1x2 figure
fig, axs = plt.subplots(
    2, 4, figsize=(12, 5), sharey=True, sharex=True, constrained_layout=True, dpi=300
)

# --- GP (Without Multi-fidelity) ---
ax_mean = axs[0, 2].imshow(
    mean_gp.view(image_dim, image_dim),
    extent=(0, 1, 0, 1),
    origin="lower",
    aspect="equal",
    vmin=mean_vmin,
    vmax=mean_vmax,
)
ax_std = axs[1, 2].imshow(
    std_gp.view(image_dim, image_dim).log10(),
    extent=(0, 1, 0, 1),
    origin="lower",
    aspect="equal",
    vmin=std_vmin,
    vmax=std_vmax,
)

# --- GP (With Multi-fidelity) ---
axs[0, 3].imshow(
    mean_migp.view(image_dim, image_dim),
    extent=(0, 1, 0, 1),
    origin="lower",
    aspect="equal",
    vmin=mean_vmin,
    vmax=mean_vmax,
)
axs[1, 3].imshow(
    std_migp.view(image_dim, image_dim).log10(),
    extent=(0, 1, 0, 1),
    origin="lower",
    aspect="equal",
    vmin=std_vmin,
    vmax=std_vmax,
)

# Add colorbar for row 0 (mean)
cbar_ax_mean = fig.add_axes([1, 0.56, 0.015, 0.34])  # [left, bottom, width, height]
cbar_mean = fig.colorbar(ax_mean, cax=cbar_ax_mean)
cbar_mean.set_label(r"$w(a) - \hat{w}(a)$")

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
# ---
# ---


# # Final layout
# plt.tight_layout()
# plt.savefig("twoDNonLinear_comparison.png", dpi=300)
# plt.close()
