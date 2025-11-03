# scripts/twoDNonLinear/plot.py
"""
Plot HF mean removed 2D fields.
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use("../journal.mplstyle")

# Begin user input
n_train_samples_list = [50, 100, 500, 1000, 1500]
n_val_samples = 500
sigma_factor = 6.0  # Number of standard deviations for the error bars
plot_idx = {"automatic": True, "idx": 6}  # Set to True for automatic index selection
plot_error = True  # Whether to plot error or not
# End user input


def load_data(n_train_samples):
    nt = n_train_samples
    nv = n_val_samples
    file_flora = f"twodnonlinear_n_train_{nt}_n_val_{nv}_results_flora.pt"
    file_floral = f"twodnonlinear_n_train_{nt}_n_val_{nv}_results_floral.pt"
    print("flora file: ", file_flora)
    print("floral file: ", file_floral)
    print("n_train_samples: ", n_train_samples)
    print("n_val_samples: ", n_val_samples)
    data_flora = torch.load(file_flora, weights_only=False)
    data_floral = torch.load(file_floral, weights_only=False)
    print(
        "Number of samples for UQ (Flora): "
        f"{data_flora['HF_field_prediction_plot'].shape[1]}"
    )
    print(
        "Number of samples for UQ (Floral): "
        f"{data_floral['HF_field_prediction_plot'].shape[1]}"
    )
    print("--" * 10)

    return data_flora, data_floral


def find_best_idx(true, prediction):
    """
    Find the index with smallest error between true and predicted values.
    """

    mean_prediction = prediction.mean(dim=1, keepdim=True)
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
    HF_field = data_floral["HF_field_plot"][idx]
    LF_field = data_floral["LF_field_plot"][idx]
    HF_field_prediction_flora = data_flora["HF_field_prediction_plot"][idx]
    HF_field_prediction_floral = data_floral["HF_field_prediction_plot"][idx]
    # field_domain = data_floral["domain_dict"]["field"]

    mean_flora = HF_field_prediction_flora.mean(dim=0)
    mean_floral = HF_field_prediction_floral.mean(dim=0)

    std_flora = HF_field_prediction_flora.std(dim=0)
    std_floral = HF_field_prediction_floral.std(dim=0)

    plot_order = ["Low-fidelity", "FLORA", "FLORAL"]

    means = {
        "Low-fidelity": LF_field,
        "FLORA": mean_flora,
        "FLORAL": mean_floral,
    }
    stds = {
        "Low-fidelity": torch.ones_like(LF_field) * 1e-6,
        "FLORA": std_flora,
        "FLORAL": std_floral,
    }

    # compute vmin and vmax for mean and std plots
    if plot_error:
        rel_errors = [
            ((HF_field - means[method]) / HF_field).abs() for method in plot_order
        ]
        mean_vmin = min(re.min().item() for re in rel_errors)
        mean_vmax = max(re.max().item() for re in rel_errors)
    else:
        mean_vmin = min(means[method].min().item() for method in plot_order)
        mean_vmax = max(means[method].max().item() for method in plot_order)

    std_vmin = min(stds[method].log10().min().item() for method in plot_order)
    std_vmax = max(stds[method].log10().max().item() for method in plot_order)

    fig, axs = plt.subplots(
        2,
        len(plot_order),
        figsize=(10, 5),
        sharey=True,
        sharex=True,
        constrained_layout=True,
        dpi=300,
    )
    for ii, method in enumerate(plot_order):
        assert method in means, f"Method {method} unavailable in means."
        assert method in stds, f"Method {method} unavailable in stds."

        # plot mean or error
        if plot_error:
            field_to_plot = ((HF_field - means[method]) / HF_field).abs()
        else:
            field_to_plot = means[method]
        field_to_plot = field_to_plot.squeeze(0)
        # print(
        #     f"min and max of {method} mean/error: "
        #     f"{field_to_plot.min().item()}, {field_to_plot.max().item()}"
        # )

        axs[0, ii].imshow(
            field_to_plot,
            extent=(0, 1, 0, 1),
            origin="lower",
            aspect="equal",
            interpolation="bilinear",
            vmin=mean_vmin,
            vmax=mean_vmax,
        )
        axs[0, ii].grid(False, which="both")
        axs[0, ii].xaxis.grid(False, which="both")
        axs[0, ii].yaxis.grid(False, which="both")
        axs[0, ii].set_title(method)
        axs[0, ii].set_xlabel(r"$x_w$")
        axs[0, ii].set_ylabel(r"$y_w$")
        axs[0, ii].label_outer()

        # plot log_10 std
        std_to_plot = stds[method].squeeze(0).log10()
        axs[1, ii].imshow(
            std_to_plot,
            extent=(0, 1, 0, 1),
            origin="lower",
            aspect="equal",
            interpolation="bilinear",
            vmin=std_vmin,
            vmax=std_vmax,
        )
        axs[1, ii].grid(False, which="both")
        axs[1, ii].xaxis.grid(False, which="both")
        axs[1, ii].yaxis.grid(False, which="both")
        axs[1, ii].set_xlabel(r"$x_w$")
        axs[1, ii].set_ylabel(r"$y_w$")
        axs[1, ii].label_outer()

    # add colorbars
    cbar_ax_mean = fig.add_axes([1, 0.56, 0.015, 0.34])  # [left, bottom, width, height]
    cbar_mean = fig.colorbar(axs[0, -1].images[0], cax=cbar_ax_mean)
    (
        cbar_mean.set_label(r"$\left|\frac{w - \hat{w}}{w}\right|$")
        if plot_error
        else cbar_mean.set_label(r"$\hat{w}$")
    )
    cbar_ax_std = fig.add_axes([1, 0.11, 0.015, 0.34])  # adjust to match lower row
    cbar_std = fig.colorbar(axs[1, -1].images[0], cax=cbar_ax_std)
    cbar_std.set_label(r"$\log_{10}(\sigma)$", labelpad=10)

    plt.savefig(f"twoDNonLinear_ntrain_{n_train_samples}_comparison.png")
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
        error_df.groupby("Method", observed=False)["L2 Error"]
        .mean()
        .to_frame()
        .join(
            uncertainty_df.groupby("Method", observed=False)["Mean Std"]
            .mean()
            .to_frame()
        )
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
    # ax.set_ylim(1e-1, 1e2)

    # Optional: make legend cleaner
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel("Mean Predictive Uncertainty (Mean Std)")
    ax.set_ylabel("Mean Predictive Error (L2 Norm)")
    plt.tight_layout()
    plt.savefig("twoDNonLinear_pareto_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # plot field
    plot_field(n_train_samples=n_train_samples_list[0])
    # plot pareto front
    # plot_pareto()
