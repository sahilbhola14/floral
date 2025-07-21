import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use("../journal.mplstyle")  # Optional style

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

n_samples = len(field.get("Prediction"))

# check that all fields have the same number of samples
assert n_samples == len(field_mfFlow.get("Prediction"))
assert n_samples == len(field_gp.get("Prediction")["mean"])
assert n_samples == len(field_gp_mfFlow.get("Prediction")["mean"])

# Ordered list of methods
method_order = ["FLORA", "FLOREN", "GP", "REGP"]

# Containers for results
error_rows = []
uncertainty_rows = []

print(f"Computing stats for {n_samples} samples...")
for ii in range(n_samples):
    HF_field = field_mfFlow.get("HF_field")[ii]

    # Predictions and stds
    predictions = {
        "FLORA": field.get("Prediction")[ii],
        "FLOREN": field_mfFlow.get("Prediction")[ii],
        "GP": field_gp.get("Prediction")["mean"][ii],
        "REGP": field_gp_mfFlow.get("Prediction")["mean"][ii],
    }
    stds = {
        "FLORA": field.get("Prediction")[ii].std(dim=0).mean().item(),
        "FLOREN": field_mfFlow.get("Prediction")[ii].std(dim=0).mean().item(),
        "GP": field_gp.get("Prediction")["std"][ii].mean().item(),
        "REGP": field_gp_mfFlow.get("Prediction")["std"][ii].mean().item(),
    }

    # Means
    means = {
        "FLORA": predictions["FLORA"].mean(dim=0),
        "FLOREN": predictions["FLOREN"].mean(dim=0),
        "GP": predictions["GP"],
        "REGP": predictions["REGP"],
    }

    for method in method_order:
        l2 = torch.norm(means[method] - HF_field).item()
        error_rows.append({"Method": method, "L2 Error": l2})
        uncertainty_rows.append({"Method": method, "Mean Std": stds[method]})

# Convert to DataFrames and set method as ordered category
error_df = pd.DataFrame(error_rows)
uncertainty_df = pd.DataFrame(uncertainty_rows)

cat_type = pd.CategoricalDtype(categories=method_order, ordered=True)
error_df["Method"] = error_df["Method"].astype(cat_type)
uncertainty_df["Method"] = uncertainty_df["Method"].astype(cat_type)

# --------------------
# Plotting
# --------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# Violin plot for L2 Errors
sns.violinplot(
    data=error_df, x="Method", y="L2 Error", inner="box", palette="Set2", ax=axs[0]
)
sns.stripplot(
    data=error_df,
    x="Method",
    y="L2 Error",
    color="k",
    size=2,
    alpha=0.7,
    jitter=True,
    ax=axs[0],
)
# axs[0].set_title("L2 Error (Prediction Accuracy)")
axs[0].grid(True, linestyle="--", alpha=0.4)

# Violin plot for Predictive Uncertainty
sns.violinplot(
    data=uncertainty_df,
    x="Method",
    y="Mean Std",
    inner="box",
    palette="Set2",
    ax=axs[1],
)
sns.stripplot(
    data=uncertainty_df,
    x="Method",
    y="Mean Std",
    color="k",
    size=2,
    alpha=0.7,
    jitter=True,
    ax=axs[1],
)
# axs[1].set_title("Predictive Uncertainty (Mean Std)")
axs[1].grid(True, linestyle="--", alpha=0.4)

# plt.suptitle("Comparison of Accuracy and Predictive Uncertainty", fontsize=14)
plt.tight_layout()
plt.savefig("oneDCorr_L2_and_Uncertainty_violin.png", dpi=300)
plt.close()
