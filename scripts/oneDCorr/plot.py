import torch
import matplotlib.pyplot as plt

plt.style.use("../journal.mplstyle")

data = torch.load("oneDCorr_10_samples_results.pt")
data_mfFlow = torch.load("oneDCorr_10_samples_results_mfFlow.pt")

data_gp = torch.load("oneDCorr_10_samples_GP_results.pt")
data_gp_mfFlow = torch.load("oneDCorr_10_samples_GP_results_mfFlow.pt")

field = data["field"]
field_mfFlow = data_mfFlow["field"]
field_gp = data_gp["field"]
field_gp_mfFlow = data_gp_mfFlow["field"]

domain = data_mfFlow["domain"].ravel()

fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
ax = ax.flatten()
for ii in range(6):
    LF_field_mfFlow = field_mfFlow.get("LF_field")[ii]
    HF_field_mfFlow = field_mfFlow.get("HF_field")[ii]
    # With ProMiNO
    Prediction_mfFlow = field_mfFlow.get("Prediction")[ii]
    mean_Prediction_mfFlow = Prediction_mfFlow.mean(dim=0)
    std_Prediction_mfFlow = Prediction_mfFlow.std(dim=0)
    # Without ProMiNO
    Prediction = field.get("Prediction")[ii]
    mean_Prediction = Prediction.mean(dim=0)
    std_Prediction = Prediction.std(dim=0)

    # Prediction GP
    Prediction_gp = field_gp.get("Prediction")
    mean_Prediction_gp = Prediction_gp.get("mean")[ii]
    std_Prediction_gp = Prediction_gp.get("std")[ii]

    # Prediction GP with Residual Learning
    Prediction_gp_mfFlow = field_gp_mfFlow.get("Prediction")
    mean_Prediction_gp_mfFlow = Prediction_gp_mfFlow.get("mean")[ii]
    std_Prediction_gp_mfFlow = Prediction_gp_mfFlow.get("std")[ii]

    ax[ii].plot(domain, LF_field_mfFlow, label="Low-fidelity", color="grey", alpha=0.5)
    ax[ii].plot(domain, HF_field_mfFlow, label="High-fidelity", color="k")

    ax[ii].plot(
        domain, mean_Prediction_mfFlow, label="ProMiNO", color="red", linestyle="--"
    )
    ax[ii].plot(domain, mean_Prediction, label="ProNO", color="green", linestyle="--")
    ax[ii].plot(domain, mean_Prediction_gp, label="GP", color="orange", linestyle="--")
    ax[ii].plot(
        domain,
        mean_Prediction_gp_mfFlow,
        label="MiGP",
        color="purple",
        linestyle="--",
    )

    ax[ii].fill_between(
        domain,
        mean_Prediction_mfFlow - std_Prediction_mfFlow,
        mean_Prediction_mfFlow + std_Prediction_mfFlow,
        color="red",
        alpha=0.2,
        label="__nolenged__",
        linestyle="--",
    )

    ax[ii].fill_between(
        domain,
        mean_Prediction - std_Prediction,
        mean_Prediction + std_Prediction,
        color="green",
        alpha=0.2,
        label="__nolenged__",
        linestyle="--",
    )

    ax[ii].fill_between(
        domain,
        mean_Prediction_gp - std_Prediction_gp,
        mean_Prediction_gp + std_Prediction_gp,
        color="orange",
        alpha=0.2,
        label="__nolenged__",
        linestyle="--",
    )

    ax[ii].fill_between(
        domain,
        mean_Prediction_gp_mfFlow - std_Prediction_gp_mfFlow,
        mean_Prediction_gp_mfFlow + std_Prediction_gp_mfFlow,
        color="purple",
        alpha=0.2,
        label="__nolenged__",
        linestyle="--",
    )

    ax[ii].set_xlabel(r"$x$")
    ax[ii].set_ylabel(r"$w(a)$")
    if ii == 0:
        ax[ii].legend(ncol=2)
plt.tight_layout()
plt.savefig("oneDCorr_results_mfFlow.png")
