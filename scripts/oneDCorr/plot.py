import torch
import matplotlib.pyplot as plt

plt.style.use("../journal.mplstyle")

data = torch.load("oneDCorr_results_mfFlow.pt")
field = data["field"]
residual = data["residual"]
domain = data["domain"].ravel()

fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
ax = ax.flatten()
for ii in range(6):
    LF_field = field.get("LF_field")[ii]
    HF_field = field.get("HF_field")[ii]
    Prediction = field.get("Prediction")[ii]
    mean_Prediction = Prediction.mean(dim=0)
    std_Prediction = Prediction.std(dim=0)
    ax[ii].plot(domain, LF_field, label="Low-fidelity", color="blue")
    ax[ii].plot(domain, HF_field, label="High-fidelity", color="k")
    ax[ii].plot(domain, mean_Prediction, label="ProMiNO", color="red", linestyle="--")
    ax[ii].fill_between(
        domain,
        mean_Prediction - std_Prediction,
        mean_Prediction + std_Prediction,
        color="red",
        alpha=0.2,
        label=r"$\pm \sigma$",
        linestyle="--",
    )
    ax[ii].set_xlabel(r"$x$")
    ax[ii].set_ylabel(r"$w(a)$")
    if ii == 0:
        ax[ii].legend()
plt.tight_layout()
plt.savefig("oneDCorr_results_mfFlow.png")
