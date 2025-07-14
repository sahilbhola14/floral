import torch
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("../journal.mplstyle")  # Or comment this line if not needed

# Load prediction data
data = torch.load("oneDCorr_10_samples_results.pt")
data_mfFlow = torch.load("oneDCorr_10_samples_results_mfFlow.pt")
data_gp = torch.load("oneDCorr_10_samples_GP_results.pt")
data_gp_mfFlow = torch.load("oneDCorr_10_samples_GP_results_mfFlow.pt")

field = data["field"]
field_mfFlow = data_mfFlow["field"]
field_gp = data_gp["field"]
field_gp_mfFlow = data_gp_mfFlow["field"]

n_samples = len(field.get("Prediction"))

# Containers for errors and uncertainty
error_data = {"Method": [], "L2 Error": []}
uncertainty_data = {"Method": [], "Mean Std": []}

for ii in range(n_samples):
    HF_field = field_mfFlow.get("HF_field")[ii]

    # Predictive means
    pred_promino = field_mfFlow.get("Prediction")[ii].mean(dim=0)
    pred_prono = field.get("Prediction")[ii].mean(dim=0)
    pred_gp = field_gp.get("Prediction")["mean"][ii]
    pred_migp = field_gp_mfFlow.get("Prediction")["mean"][ii]

    # Compute L2 Errors
    error_data["Method"] += ["ProMiNO", "ProNO", "GP", "MiGP"]
    error_data["L2 Error"] += [
        torch.norm(pred_promino - HF_field).item(),
        torch.norm(pred_prono - HF_field).item(),
        torch.norm(pred_gp - HF_field).item(),
        torch.norm(pred_migp - HF_field).item(),
    ]

    # Compute mean predictive std (averaged over domain)
    std_promino = field_mfFlow.get("Prediction")[ii].std(dim=0).mean().item()
    std_prono = field.get("Prediction")[ii].std(dim=0).mean().item()
    std_gp = field_gp.get("Prediction")["std"][ii].mean().item()
    std_migp = field_gp_mfFlow.get("Prediction")["std"][ii].mean().item()

    uncertainty_data["Method"] += ["ProMiNO", "ProNO", "GP", "MiGP"]
    uncertainty_data["Mean Std"] += [std_promino, std_prono, std_gp, std_migp]

# --------------------
# Plotting
# --------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# Violin plot for L2 Errors
sns.violinplot(
    data=error_data, x="Method", y="L2 Error", inner="box", palette="Set2", ax=axs[0]
)
sns.stripplot(
    data=error_data,
    x="Method",
    y="L2 Error",
    color="k",
    size=4,
    alpha=0.7,
    jitter=True,
    ax=axs[0],
)
axs[0].set_title("L2 Error (Prediction Accuracy)")
axs[0].grid(True, linestyle="--", alpha=0.4)

# Violin plot for Predictive Uncertainty
sns.violinplot(
    data=uncertainty_data,
    x="Method",
    y="Mean Std",
    inner="box",
    palette="Set2",
    ax=axs[1],
)
sns.stripplot(
    data=uncertainty_data,
    x="Method",
    y="Mean Std",
    color="k",
    size=4,
    alpha=0.7,
    jitter=True,
    ax=axs[1],
)
axs[1].set_title("Predictive Uncertainty (Mean Std)")
axs[1].grid(True, linestyle="--", alpha=0.4)

plt.suptitle("Comparison of Accuracy and Predictive Uncertainty", fontsize=14)
plt.tight_layout()
plt.savefig("oneDCorr_L2_and_Uncertainty_violin.png", dpi=300)
plt.show()
