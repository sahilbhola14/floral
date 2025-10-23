import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use("../journal.mplstyle")

# Begin user input
n_train_samples_list = [100, 500, 1000, 1500]
n_val_samples = 500
sigma_factor = 6.0  # Number of standard deviations for the error bars
plot_idx = {"automatic": False, "idx": 6}  # Set to True for automatic index selection
# End user input


def load_data(n_train_samples):
    file_flora = (
        f"onedcorr_n_train_{n_train_samples}_n_val_{n_val_samples}_results_flora.pt"
    )
    file_floral = (
        f"onedcorr_n_train_{n_train_samples}_n_val_{n_val_samples}_results_floral.pt"
    )
    print("flora file: ", file_flora)
    print("floral file: ", file_floral)
    print("n_train_samples: ", n_train_samples)
    print("n_val_samples: ", n_val_samples)
    data_flora = torch.load(file_flora, weights_only=False)
    data_floral = torch.load(file_floral, weights_only=False)

    return data_flora, data_floral


def find_best_idx(true, prediction):
    """
    Find the index with smallest error between true and predicted values.
    """
    mean_prediction = prediction.mean(dim=1)
    error = (true - mean_prediction).abs().flatten(start_dim=1).mean(dim=1)
    idx = error.argmin().item()
    print(f"Best index found: {idx} with error {error.min().item()}")
    return idx


def plot_field(n_train_samples):

    data_flora, data_floral = load_data(n_train_samples)

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

    plt.savefig("oneDCorr_field_comparison.png", dpi=300)
    plt.close()


def get_pareto_data(n_train_samples):
    data_flora, data_floral = load_data(n_train_samples)

    HF_field = data_floral["HF_field_plot"]  # (batch_size, channels, *dims)
    LF_field = data_floral["LF_field_plot"]  # (batch_size, channels, *dims)
    HF_field_prediction_flora = data_flora[
        "HF_field_prediction_plot"
    ]  # (batch_size, n_gen, channels, *dims)
    HF_field_prediction_floral = data_floral[
        "HF_field_prediction_plot"
    ]  # (batch_size, n_gen, channels, *dims)

    # means
    means = {
        "FLORA": HF_field_prediction_flora.mean(dim=1),
        "FLORAL": HF_field_prediction_floral.mean(dim=1),
    }
    # stds
    stds = {
        "FLORA": HF_field_prediction_flora.std(dim=1),
        "FLORAL": HF_field_prediction_floral.std(dim=1),
    }

    # errors
    errors = {
        "FLORA": torch.norm(HF_field.flatten(1) - means["FLORA"].flatten(1), dim=1),
        "FLORAL": torch.norm(HF_field.flatten(1) - means["FLORAL"].flatten(1), dim=1),
        "Low-fidelity": torch.norm(HF_field.flatten(1) - LF_field.flatten(1), dim=1),
    }

    # uncertainty (mean std)
    uncertainty = {
        "FLORA": stds["FLORA"].flatten(1).mean(dim=1),
        "FLORAL": stds["FLORAL"].flatten(1).mean(dim=1),
        "Low-fidelity": torch.ones_like(errors["Low-fidelity"])
        * 1e-6,  # no uncertainty for LF
    }

    # Convert to rows for each method
    error_rows, uncertainty_rows = [], []
    for method in ["FLORA", "FLORAL", "Low-fidelity"]:
        for e, u in zip(errors[method].tolist(), uncertainty[method].tolist()):
            error_rows.append({"Method": method, "L2 Error": e})
            uncertainty_rows.append({"Method": method, "Mean Std": u})

    # Convert to DataFrames and set method as ordered category
    error_df = pd.DataFrame(error_rows)
    uncertainty_df = pd.DataFrame(uncertainty_rows)

    method_order = ["Low-fidelity", "FLORA", "FLORAL"]
    cat_type = pd.CategoricalDtype(categories=method_order, ordered=True)
    error_df["Method"] = error_df["Method"].astype(cat_type)
    uncertainty_df["Method"] = uncertainty_df["Method"].astype(cat_type)

    # Compute method-wise means
    summary_df = (
        error_df.groupby("Method")["L2 Error"]
        .mean()
        .to_frame()
        .join(uncertainty_df.groupby("Method")["Mean Std"].mean().to_frame())
        .reset_index()
    )

    # extract the number of samples
    assert data_flora["n_samples"] == data_floral["n_samples"]
    assert data_flora["n_train"] == data_floral["n_train"]
    assert data_flora["n_val"] == data_floral["n_val"]

    summary_df["Samples (total)"] = data_flora["n_samples"]
    summary_df["Samples (train)"] = data_flora["n_train"]
    summary_df["Samples (validation)"] = data_flora["n_val"]
    return summary_df


def plot_pareto():

    # Combine all data into one DataFrame
    all_data = []
    for n_train_samples in n_train_samples_list:
        df = get_pareto_data(n_train_samples)
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Split data
    df_lf = combined_df[combined_df["Method"] == "Low-fidelity"].iloc[
        [0]
    ]  # take one row
    df_rest = combined_df[combined_df["Method"] != "Low-fidelity"]

    # Plot methods that depend on Samples (Train)
    plt.figure(figsize=(10, 5))
    # Overlay the Low-fidelity baseline separately
    ax = sns.scatterplot(
        data=df_lf,
        x="Mean Std",
        y="L2 Error",
        color="gray",
        s=200,
        marker="*",
        edgecolor="black",
        label="Low-fidelity (reference)",
    )

    sns.scatterplot(
        data=df_rest,
        x="Mean Std",
        y="L2 Error",
        hue="Method",
        hue_order=["FLORA", "FLORAL"],
        style="Samples (train)",
        palette="Set2",
        s=100,
        edgecolor="black",
        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1e-1)
    ax.set_ylim(1e-2, 1e1)

    # Optional: make legend cleaner
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel("Mean Predictive Uncertainty (Mean Std)")
    ax.set_ylabel("Mean Predictive Accuracy (L2 Norm)")
    plt.tight_layout()
    plt.savefig("oneDCorr_pareto_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # plot field
    # plot_field(n_train_samples=n_train_samples_list[0])
    # plot pareto front
    plot_pareto()
