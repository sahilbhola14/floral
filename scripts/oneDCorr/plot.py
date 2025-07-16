import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.style.use("../journal.mplstyle")

# Begin user input
n_samples = 10
n_sensors = 100
# End user input

# Load data
data = torch.load(f"oneDCorr_{n_samples}_samples_{n_sensors}_sensors_results.pt")
data_mfFlow = torch.load(
    f"oneDCorr_{n_samples}_samples_{n_sensors}_sensors_results_mfFlow.pt"
)
data_gp = torch.load(f"oneDCorr_{n_samples}_samples_{n_sensors}_sensors_GP_results.pt")
data_gp_mfFlow = torch.load(
    f"oneDCorr_{n_samples}_samples_{n_sensors}_sensors_GP_results_mfFlow.pt"
)

field = data["field"]
field_mfFlow = data_mfFlow["field"]
field_gp = data_gp["field"]
field_gp_mfFlow = data_gp_mfFlow["field"]

domain = data_mfFlow["domain"].ravel()
ii = 6  # Select sample index

# Extract relevant data
LF_field = field_mfFlow.get("LF_field")[ii]
HF_field = field_mfFlow.get("HF_field")[ii]

Prediction_prono = field.get("Prediction")[ii]
mean_prono = Prediction_prono.mean(dim=0)
std_prono = Prediction_prono.std(dim=0)

Prediction_promino = field_mfFlow.get("Prediction")[ii]
mean_promino = Prediction_promino.mean(dim=0)
std_promino = Prediction_promino.std(dim=0)

Prediction_gp = field_gp.get("Prediction")
mean_gp = Prediction_gp["mean"][ii]
std_gp = Prediction_gp["std"][ii]

Prediction_migp = field_gp_mfFlow.get("Prediction")
mean_migp = Prediction_migp["mean"][ii]
std_migp = Prediction_migp["std"][ii]

# Set up 1x2 figure
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# --- Left subplot: LF vs HF ---
axs[0].plot(domain, LF_field, label="Low-fidelity", color="grey", alpha=0.6)
axs[0].plot(domain, HF_field, label="High-fidelity", color="k")
# axs[0].set_title("Low vs High Fidelity")
axs[0].set_xlabel(r"$x$")
axs[0].set_ylabel(r"$w(a)$")
axs[0].legend()

# --- Right subplot: Methods ---
ax_main = axs[1]
ax_main.plot(domain, HF_field, label="High-fidelity", color="k")

ax_main.plot(domain, mean_prono, label="FLORA", color="green", linestyle="--")
ax_main.fill_between(
    domain, mean_prono - std_prono, mean_prono + std_prono, color="green", alpha=0.2
)

ax_main.plot(domain, mean_promino, label="FLOREN", color="red", linestyle="--")
ax_main.fill_between(
    domain,
    mean_promino - std_promino,
    mean_promino + std_promino,
    color="red",
    alpha=0.2,
)

ax_main.plot(domain, mean_gp, label="GP", color="orange", linestyle="--")
ax_main.fill_between(
    domain, mean_gp - std_gp, mean_gp + std_gp, color="orange", alpha=0.2
)

ax_main.plot(domain, mean_migp, label="REGP", color="blue", linestyle="--")
ax_main.fill_between(
    domain, mean_migp - std_migp, mean_migp + std_migp, color="blue", alpha=0.2
)

# ax_main.set_title("Method Comparison")
ax_main.set_xlabel(r"$x$")
ax_main.legend(ncol=2)

# --- Inset Zoom-in on Right Plot ---

# Create inset axes
# axins = inset_axes(ax_main, width="30%", height="50%", loc="best", borderpad=3)
# axins = inset_axes(ax_main, width="30%", height="50%", loc="center", borderpad=3)
axins = inset_axes(
    ax_main,
    width="20%",  # or float (absolute units)
    height="40%",
    bbox_to_anchor=(
        0.32,
        0.0000,
        1,
        1,
    ),  # (x0, y0, width, height) in axes coords of ax_main
    bbox_transform=ax_main.transAxes,
    borderpad=1.5,
    loc="lower left"  # loc here positions the bbox_to_anchor box relative to loc,
    # but you can just keep loc='center' to place bbox_to_anchor exactly
)

# Define zoom range (you can change this to a region of interest)
x1, x2 = 0.8, 1.0
y1, y2 = -0.5, 1.5
# y1, y2 = (
#     HF_field[(domain > x1) & (domain < x2)].min(),
#     HF_field[(domain > x1) & (domain < x2)].max(),
# )
y_margin = 0.05 * (y2 - y1)
y1 -= y_margin
y2 += y_margin

# Same plots inside the inset
axins.plot(domain, HF_field, color="k")
axins.plot(domain, mean_prono, color="green", linestyle="--")
axins.fill_between(
    domain, mean_prono - std_prono, mean_prono + std_prono, color="green", alpha=0.2
)

axins.plot(domain, mean_promino, color="red", linestyle="--")
axins.fill_between(
    domain,
    mean_promino - std_promino,
    mean_promino + std_promino,
    color="red",
    alpha=0.2,
)

axins.plot(domain, mean_gp, color="orange", linestyle="--")
axins.fill_between(
    domain, mean_gp - std_gp, mean_gp + std_gp, color="orange", alpha=0.2
)

axins.plot(domain, mean_migp, color="blue", linestyle="--")
axins.fill_between(
    domain, mean_migp - std_migp, mean_migp + std_migp, color="blue", alpha=0.2
)

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.tick_params(labelsize=8)

# Connect inset to main plot
mark_inset(ax_main, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1)

# Final layout
plt.tight_layout()
plt.savefig("oneDCorr_comparison.png", dpi=300)
plt.close()
