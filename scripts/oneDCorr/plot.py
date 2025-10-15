import torch
import matplotlib.pyplot as plt

plt.style.use("../journal.mplstyle")

# Begin user input
n_samples = 2000
sigma_factor = 4.0  # Number of standard deviations for the error bars
plot_idx = {"automatic": True, "idx": 6}  # Set to True for automatic index selection
# End user input
file_floral = f"oneDCorr_{n_samples}_samples_results_floral.pt"
file_flora = f"oneDCorr_{n_samples}_samples_results_flora.pt"
print(f"Loading results from {file_floral} and {file_flora} with {n_samples} samples.")


def find_best_idx(true, prediction):
    """
    Find the index with smallest error between true and predicted values.
    """
    mean_prediction = prediction.mean(dim=1)
    error = (true - mean_prediction).abs().flatten(start_dim=1).mean(dim=1)
    idx = error.argmin().item()
    print(f"Best index found: {idx} with error {error.min().item()}")
    return idx


def plot_field(data_flora, data_floral):
    if plot_idx["automatic"]:
        idx = find_best_idx(
            true=data_floral["HF_field_plot"],
            prediction=data_floral["HF_field_prediction_plot"],
        )
    else:
        idx = plot_idx["idx"]
    print(f"Plotting index {idx}")

    # extract data
    HF_field = data_floral["HF_field_plot"][idx].ravel()
    LF_field = data_floral["LF_field_plot"][idx].ravel()
    HF_field_prediction_flora = data_flora["HF_field_prediction_plot"][idx]
    HF_field_prediction_floral = data_floral["HF_field_prediction_plot"][idx]
    field_domain = data_floral["domain_dict"]["field"].ravel()

    mean_flora = HF_field_prediction_flora.mean(dim=0).ravel()
    mean_floral = HF_field_prediction_floral.mean(dim=0).ravel()

    std_flora = HF_field_prediction_flora.std(dim=0).ravel()
    std_floral = HF_field_prediction_floral.std(dim=0).ravel()

    fig, axs = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True, sharey=True, sharex=True
    )
    axs[0].plot(field_domain, HF_field, label="High-fidelity", color="black")
    axs[0].plot(
        field_domain, LF_field, label="Low-fidelity", color="gray", linestyle="dashed"
    )

    axs[1].plot(field_domain, HF_field, label="High-fidelity", color="black")
    axs[1].plot(
        field_domain, mean_flora, label=r"$\mu_{FLORA}$", color="green", linestyle="--"
    )
    axs[1].fill_between(
        field_domain,
        mean_flora - sigma_factor * std_flora,
        mean_flora + sigma_factor * std_flora,
        color="green",
        alpha=0.3,
        label="",
    )
    axs[1].plot(
        field_domain, mean_floral, label=r"$\mu_{FLORAL}$", color="red", linestyle=":"
    )
    axs[1].fill_between(
        field_domain,
        mean_floral - sigma_factor * std_floral,
        mean_floral + sigma_factor * std_floral,
        color="red",
        alpha=0.3,
        label="",
    )

    for ax in axs:
        ax.set_xlabel(r"$x_w$")
        ax.set_ylabel(r"$w(x_w)$")
        ax.legend(loc="upper right")
        ax.grid()
        ax.label_outer()

    plt.savefig("oneD_field_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    data_flora = torch.load(file_flora, weights_only=False)
    data_floral = torch.load(file_floral, weights_only=False)
    # plot field
    plot_field(data_flora, data_floral)
