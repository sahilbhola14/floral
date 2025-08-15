import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use("../journal.mplstyle")  # Optional style

# Begin user input
n_samples = 50
n_sensors = 20
# End user input

# Load data
data = torch.load(
    f"oneDCorr_{n_samples}_samples_{n_sensors}_sensors_results.pt", weights_only=False
)
data_mfFlow = torch.load(
    f"oneDCorr_{n_samples}_samples_{n_sensors}_sensors_results_mfFlow.pt",
    weights_only=False,
)
data_gp = torch.load(
    f"oneDCorr_{n_samples}_samples_{n_sensors}_sensors_GP_results.pt",
    weights_only=False,
)
data_gp_mfFlow = torch.load(
    f"oneDCorr_{n_samples}_samples_{n_sensors}_sensors_GP_results_mfFlow.pt",
    weights_only=False,
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
method_order = ["FLORA", "FLOREN", "GP", "REGP", "Low-fidelity"]

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
        "Low-fidelity": field.get("LF_field")[ii],
    }
    stds = {
        "FLORA": field.get("Prediction")[ii].std(dim=0).mean().item(),
        "FLOREN": field_mfFlow.get("Prediction")[ii].std(dim=0).mean().item(),
        "GP": field_gp.get("Prediction")["std"][ii].mean().item(),
        "REGP": field_gp_mfFlow.get("Prediction")["std"][ii].mean().item(),
        "Low-fidelity": 0.0,
    }

    # Means
    means = {
        "FLORA": predictions["FLORA"].mean(dim=0),
        "FLOREN": predictions["FLOREN"].mean(dim=0),
        "GP": predictions["GP"],
        "REGP": predictions["REGP"],
        "Low-fidelity": predictions["Low-fidelity"],
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

# --------------------------------------------
# Plot L2 Error vs. Predictive Uncertainty
# --------------------------------------------

# Step 1: Compute method-wise means
summary_df = (
    error_df.groupby("Method")["L2 Error"]
    .mean()
    .to_frame()
    .join(uncertainty_df.groupby("Method")["Mean Std"].mean().to_frame())
    .reset_index()
)

# Step 2: Plot
plt.figure(figsize=(6, 5))
ax = sns.scatterplot(
    data=summary_df,
    x="Mean Std",
    y="L2 Error",
    hue="Method",
    palette="Set2",
    s=100,
    edgecolor="black",
)

ax.legend(title="", loc="upper right")

# Annotate each point with method name
label_offset_x = 0.01
label_offset_y = 0.005

for _, row in summary_df.iterrows():
    plt.text(
        row["Mean Std"] + label_offset_x,
        row["L2 Error"] + label_offset_y,
        row["Method"],
        fontsize=9,
        weight="bold",
    )

plt.xlabel("Mean Std")
plt.ylabel("L2 Error")
# plt.title("L2 Error vs Predictive Uncertainty")
plt.grid(True, linestyle="--", alpha=0.5)
plt.yscale("log")
plt.ylim(1e-2, 1e1)
plt.xlim(right=0.2)
plt.tight_layout()
plt.savefig("L2_vs_Uncertainty.png", dpi=300)
plt.close()
