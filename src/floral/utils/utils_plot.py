"""
Plotting utilities
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from abc import abstractmethod, ABC

# Global color palette shared across all plot classes
_METHOD_COLORS = {
    "High-fidelity": "#000000",
    "Low-fidelity": "#999999",
    "FLORA": "#E10000",
    "FLORAL": "#0040FF",
    "FNO": "#008000",
    "Residual FNO": "#FF8C00",
}


def find_best_idx(true, pred):
    """Find the best sample index based on L1 error"""
    assert true.shape == pred.shape
    # compute error for each sample (n_samples,)
    error = (true - pred).abs().flatten(1).mean(1)
    idx_best = error.argmin().item()
    print(f"Best sample idx: {idx_best}, error: {error[idx_best]:.4e}")
    return idx_best


def compute_RMSE(true, pred):
    """Root Mean Squared Error"""
    assert pred.shape == true.shape
    return torch.sqrt(torch.mean((pred - true) ** 2))


def compute_NRMSE(true, pred):
    """Normalized RMSE = ||u_pred - u_true||_2 / ||u_true||_2.
    normalized RMSE (ensuring scale independence)
    """
    assert pred.shape == true.shape
    num = torch.norm(pred - true, p=2)
    den = torch.norm(true, p=2)
    return num / (den + 1e-12)


def compute_CRMSE(true, pred):
    """
    Conserved RMSE = || sum(u_pred) - sum(u_true) ||_2 / N
    RMSE of conserved value (deviation from conserved physical quantity)
    """
    assert pred.shape == true.shape
    batch_size = len(pred)
    diff = torch.sum(pred, dim=list(range(1, pred.ndim))) - torch.sum(
        true, dim=list(range(1, true.ndim))
    )
    return torch.norm(diff, p=2) / batch_size


def print_error_summary(error_dict):
    """Print RMSE, CRMSE, and NRMSE for all models in a DataFrame format."""
    metrics = ["RMSE", "CRMSE", "NRMSE"]
    missing_keys = [k for k in metrics if k not in error_dict.keys()]
    assert len(missing_keys) == 0, f"Missing keys: {', '.join(missing_keys)}"
    data = {}
    for metric in metrics:
        metric_dict = error_dict.get(metric, {})
        data[metric] = {
            k: float(v.item() if hasattr(v, "item") else v)
            for k, v in metric_dict.items()
        }
    df = pd.DataFrame(data).round(6)
    print(df)


class BasePlot(ABC):
    @abstractmethod
    def make_field_sample_plot(self):
        """plot mean fields"""
        pass

    @abstractmethod
    def make_error_sample_plot(self):
        pass


class BaseResidual(ABC):
    def __init__(self, data_flora, data_floral):
        self.data_flora = data_flora
        self.data_floral = data_floral

        # High-fidelity data (B, channels, *dims)
        self.HF_field = self.data_floral["HF_field"]
        # Low-fidelity data (B, channels, *dims)
        self.LF_field = self.data_floral["LF_field"]
        # Prediction Flora
        self.HF_field_prediction_flora = self.data_flora["prediction"]
        # Prediction Floral
        self.HF_field_prediction_floral = self.data_floral["prediction"]
        # condition
        self.full_condition = self.data_floral["condition"]
        # extract num train and val
        self.n_train = self.data_floral["n_train"]
        self.n_val = self.data_floral["n_val"]
        self.n_test = self.data_floral["n_test"]
        self.n_samples = self.n_train + self.n_val + self.n_test

        self.n_avail_samples = len(self.HF_field)

        # domains
        self.field_domain = self.data_flora["domain_dict"]["field"][0]

        self.n_avail_samples = len(self.HF_field)
        assert len(self.LF_field) == self.n_avail_samples
        assert len(self.HF_field_prediction_flora) == self.n_avail_samples
        assert len(self.HF_field_prediction_floral) == self.n_avail_samples

        # number of UQ samples
        self.n_gen = self.HF_field_prediction_floral.shape[1]
        assert self.n_gen == self.HF_field_prediction_flora.shape[1]

        # compute mean and std over the UQ samples (B, C, *dims)
        self.mean_dict = {
            "Low-fidelity": self.LF_field,
            "High-fidelity": self.HF_field,
            "FLORA": self.HF_field_prediction_flora.mean(1),
            "FLORAL": self.HF_field_prediction_floral.mean(1),
        }

        self.std_dict = {
            "Low-fidelity": torch.ones_like(self.LF_field) * 1e-6,
            "FLORA": self.HF_field_prediction_flora.std(1, correction=0),
            "FLORAL": self.HF_field_prediction_floral.std(1, correction=0),
        }

    def __call__(self, verbose: bool = False, model: str = None):
        """compute the residual norm"""
        df = []
        for k in self.mean_dict.keys():
            prediction = self.mean_dict.get(k).squeeze(1)
            condition = self.condition
            # compute the residual for the oneDcorr
            residual = self.comp_residual(
                prediction=prediction, condition=condition, domain=self.field_domain
            )
            # compute the residual norm
            residual_norm = torch.norm(
                residual, dim=tuple(range(1, prediction.ndim))
            ).mean()
            # update dictionary
            df.append(
                {
                    "Model": model,
                    "Method": k,
                    "Samples (train)": self.n_train,
                    "Samples (val)": self.n_val,
                    "Samples (total)": self.n_samples,
                    "Residual Norm": float(residual_norm),
                }
            )
        df = pd.DataFrame(df)
        if verbose:
            print(
                f"\n=== Residual summary for n_train: {self.n_train} "
                f"and n_val: {self.n_val} ==="
            )
            print(df)
        return df

    @staticmethod
    def plot_residual(combined_df, **kwargs):
        """plot the errors"""
        # Create 1×1 subplots
        figsize = kwargs.get("figsize", (4, 4))
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout="compressed")

        # extract low-fidelity residual (same for all train samples)
        subset_LF = combined_df[combined_df["Method"] == "Low-fidelity"]
        residual_LF = subset_LF["Residual Norm"].mean()

        # extract subset and build legend label
        subset = combined_df[combined_df["Method"].isin(["FLORA", "FLORAL"])].copy()

        def _make_label(r):
            if r["Model"] == "FNO":
                return "FNO" if r["Method"] == "FLORA" else "Residual FNO"
            return r["Method"]

        subset["Label"] = subset.apply(_make_label, axis=1)
        label_order = subset["Label"].unique().tolist()
        sns.scatterplot(
            data=subset,
            x="Samples (train)",
            y="Residual Norm",
            style="Label",
            style_order=label_order,
            hue="Label",
            hue_order=label_order,
            palette="Set2",
            s=150,
            edgecolor="black",
        )

        # Add horizontal LF reference line
        ax.axhline(
            residual_LF, linestyle="--", color="k", linewidth=2, label="Low-fidelity"
        )

        # ax.set_title(metric)
        ax.set_xlabel("Samples (train)")
        ax.set_ylabel(r"Mean physical residual $L_2$ norm")
        ax.legend().set_title("Model")
        ax.set_yscale("log")
        ax.set_xscale("log")
        xlim_range = kwargs.get("xlim_range", (1e1, 1e4))
        ylim_range = kwargs.get("ylim_range", (1e-3, 1e1))
        ax.set_xlim(left=xlim_range[0], right=xlim_range[1])
        ax.set_ylim(bottom=ylim_range[0], top=ylim_range[1])

        # # Annotate the LF value (formatted in scientific notation)
        # ax.text(
        #     xlim_range[0] * 1.1,          # slightly right of the left x-limit
        #     residual_LF * 0.90,           # slightly above the line
        #     f"Low-fidelity",
        #     fontsize=13,
        #     color="k"
        # )

        plt.savefig("residual_comparison.png", dpi=300, pad_inches=0.1)
        plt.close()

    @abstractmethod
    def comp_residual(self):
        """compute the residual"""
        pass


class ErrorSummary:
    def __init__(self, data_flora: dict, data_floral: dict):
        super(ErrorSummary, self).__init__()
        self.data_flora = data_flora
        self.data_floral = data_floral

        # High-fidelity data (B, n_fields_per_sample, channels, *dims)
        self.HF_field = self.data_floral["HF_field"]
        # Low-fidelity data (B, n_fields_per_sample, channels, *dims)
        self.LF_field = self.data_floral["LF_field"]
        # Prediction Flora (B, n_fields_per_sample*n_gen, channels, *dims)
        self.HF_field_prediction_flora = self.data_flora["prediction"]
        # Prediction Floral (B, n_fields_per_sample*n_gen, channels, *dims)
        self.HF_field_prediction_floral = self.data_floral["prediction"]
        # extract num train and val
        self.n_train = self.data_floral["n_train"]
        self.n_val = self.data_floral["n_val"]
        self.n_test = self.data_floral["n_test"]
        self.n_samples = self.n_train + self.n_val + self.n_test

        self.n_avail_samples = len(self.HF_field)
        assert len(self.LF_field) == self.n_avail_samples
        assert len(self.HF_field_prediction_flora) == self.n_avail_samples
        assert len(self.HF_field_prediction_floral) == self.n_avail_samples

        # number of UQ samples
        self.n_fields_per_sample = self.data_floral["n_fields_per_sample"]
        self.n_gen = (
            self.HF_field_prediction_floral.shape[1] // self.n_fields_per_sample
        )
        assert (
            self.HF_field_prediction_floral.shape[1]
            == self.n_fields_per_sample * self.n_gen
        )

        # compute mean and std over the UQ samples (B, C, *dims)
        # for Low-fidelity and High-fidelity,
        # we take the mean over the number of realizations
        self.mean_dict = {
            "Low-fidelity": self.LF_field.mean(1),
            "High-fidelity": self.HF_field.mean(1),
            "FLORA": self.HF_field_prediction_flora.mean(1),
            "FLORAL": self.HF_field_prediction_floral.mean(1),
        }

    def __call__(self, model: str, verbose: bool = False):
        error_dict = {
            "RMSE": {},
            "NRMSE": {},
            "CRMSE": {},
        }

        for k in self.mean_dict:
            true = self.mean_dict["High-fidelity"]
            pred = self.mean_dict[k]
            if k != "High-fidelity":
                # RMSE
                RMSE = compute_RMSE(true=true, pred=pred)
                error_dict["RMSE"][k] = RMSE
                # NRMSE
                NRMSE = compute_NRMSE(true=true, pred=pred)
                error_dict["NRMSE"][k] = NRMSE
                # CRMSE
                CRMSE = compute_CRMSE(true=true, pred=pred)
                error_dict["CRMSE"][k] = CRMSE

        if verbose:
            print(
                f"\n=== Error summary for n_train: {self.n_train} "
                f"and n_val: {self.n_val} ==="
            )
            print_error_summary(error_dict)

        # Flatten the nested dict with fidelity names and metrics
        df = []
        for metric, values in error_dict.items():
            for fidelity, val in values.items():
                df.append(
                    {
                        "Samples (train)": self.n_train,
                        "Samples (val)": self.n_val,
                        "Samples (total)": self.n_samples,
                        "Model": model,
                        "Method": fidelity,
                        "Metric": metric,
                        "Value": val.item() if hasattr(val, "item") else val,
                    }
                )
        return pd.DataFrame(df)

    @staticmethod
    def plot_error_vs_train_res(combined_df, **kwargs):
        """plot the errors"""
        metrics = ["RMSE", "NRMSE", "CRMSE"]

        def _make_label(r):
            if r["Model"] == "FNO":
                return "FNO" if r["Method"] == "FLORA" else "Residual FNO"
            return r["Method"]

        # extract subsets
        df_LF = combined_df[combined_df["Method"] == "Low-fidelity"]
        df_rest = combined_df[combined_df["Method"] != "Low-fidelity"].copy()
        df_rest["Label"] = df_rest.apply(_make_label, axis=1)
        label_order = df_rest["Label"].unique().tolist()

        palette = {k: _METHOD_COLORS[k] for k in label_order if k in _METHOD_COLORS}

        # Create 1×3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="compressed")
        for ii, (ax, metric) in enumerate(zip(axes, metrics)):
            # Filter dataframe for each metric
            subset_plot = df_rest[df_rest["Metric"] == metric]
            subset_plot_LF = df_LF[df_LF["Metric"] == metric]
            metric_LF = subset_plot_LF["Value"].mean()

            # Scatter plot
            sns.scatterplot(
                data=subset_plot,
                x="Resolution (train)",
                y="Value",
                style="Label",
                style_order=label_order,
                hue="Label",
                hue_order=label_order,
                palette=palette,
                s=250,
                edgecolor="black",
                ax=ax,
            )

            # Add horizontal LF reference line
            ax.axhline(
                metric_LF,
                linestyle="--",
                color=_METHOD_COLORS["Low-fidelity"],
                linewidth=2,
                label="Low-fidelity",
            )

            # Titles and labels
            ax.set_xlabel("Resolution (train)", fontsize=15)
            ax.set_ylabel(metric, fontsize=15)
            if ii == 0:
                ax.legend().set_title("Model")
            else:
                ax.legend_.remove()
            ax.set_yscale("log")
            ax.set_xscale("log")
            xlim_range = kwargs.get("xlim_range", (1e1, 1e4))
            ylim_range = kwargs.get("ylim_range", (1e-2, 1e1))
            ax.set_xlim(left=xlim_range[0], right=xlim_range[1])
            ax.set_ylim(bottom=ylim_range[0], top=ylim_range[1])

        plt.savefig("error_vs_train_resolution_comparison.png", dpi=300, pad_inches=0.1)
        plt.close()

    @staticmethod
    def plot_error(combined_df, **kwargs):
        """plot the errors"""
        metrics = ["RMSE", "NRMSE", "CRMSE"]
        titles = ["RMSE", "nRMSE", "cRMSE"]
        desired_order = ["Low-fidelity", "FNO", "Residual FNO", "FLORA", "FLORAL"]

        # extract low-fidelity residual (same for all train samples)
        subset_LF = combined_df[combined_df["Method"] == "Low-fidelity"]

        def _make_label(r):
            if r["Model"] == "FNO":
                return "FNO" if r["Method"] == "FLORA" else "Residual FNO"
            return r["Method"]

        xlim_range = kwargs.get("xlim_range", (1e1, 1e4))
        ylim_range = kwargs.get("ylim_range", (1e-2, 1e1))

        # sharey keeps ticks in sync; we hide y-axis on non-leftmost panels manually
        fig, axes = plt.subplots(
            1, 3, figsize=(12, 4), layout="compressed", sharey=True
        )
        legend_handles = None
        legend_labels = None
        for ii, (ax, metric) in enumerate(zip(axes, metrics)):
            # Filter dataframe for each metric
            subset_plot = combined_df[
                (combined_df["Metric"] == metric)
                & (combined_df["Method"] != "Low-fidelity")
            ].copy()
            subset_plot["Label"] = subset_plot.apply(_make_label, axis=1)
            label_order = [
                label
                for label in desired_order
                if label in subset_plot["Label"].unique()
            ]
            palette = {k: _METHOD_COLORS[k] for k in label_order if k in _METHOD_COLORS}
            subset_LF_plot = subset_LF[subset_LF["Metric"] == metric]
            metric_LF = subset_LF_plot["Value"].mean()

            # Scatter plot
            sns.scatterplot(
                data=subset_plot,
                x="Samples (train)",
                y="Value",
                style="Label",
                style_order=label_order,
                hue="Label",
                hue_order=label_order,
                palette=palette,
                s=250,
                edgecolor="black",
                ax=ax,
            )

            # Add horizontal LF reference line
            ax.axhline(
                metric_LF,
                linestyle="--",
                color=_METHOD_COLORS["Low-fidelity"],
                linewidth=2,
                label="Low-fidelity",
            )

            ax.set_xlabel("Samples (train)", fontsize=18)
            ax.set_title(titles[ii], fontsize=18)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlim(left=xlim_range[0], right=xlim_range[1])
            ax.set_ylim(bottom=ylim_range[0], top=ylim_range[1])

            if ii == 0:
                ax.set_ylabel("Error", fontsize=18)
                handles, labels = ax.get_legend_handles_labels()
                order_map = {label: idx for idx, label in enumerate(desired_order)}
                ordered_pairs = sorted(
                    zip(handles, labels),
                    key=lambda pair: order_map.get(pair[1], len(desired_order)),
                )
                legend_handles = [handle for handle, _ in ordered_pairs]
                legend_labels = [label for _, label in ordered_pairs]
            else:
                ax.set_ylabel("")
            if ax.legend_ is not None:
                ax.legend_.remove()

        if legend_handles is not None and legend_labels is not None:
            fig.legend(
                legend_handles,
                legend_labels,
                title="Model",
                loc="center left",
                fontsize=18,
                bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0.0,
            )

        plt.savefig(
            "error_comparison.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()


class twoDPlot(BasePlot):
    """Plot multiple samples"""

    def __init__(self, data_flora: dict, data_floral: dict):
        super(twoDPlot, self).__init__()
        self.data_flora = data_flora
        self.data_floral = data_floral

        # High-fidelity data (B, channels, *dims)
        self.HF_field = self.data_floral["HF_field"]
        # Low-fidelity data (B, channels, *dims)
        self.LF_field = self.data_floral["LF_field"]
        # Prediction Flora
        self.HF_field_prediction_flora = self.data_flora["prediction"]
        # Prediction Floral
        self.HF_field_prediction_floral = self.data_floral["prediction"]
        # extract num train and val
        self.n_train = self.data_floral["n_train"]
        self.n_val = self.data_floral["n_val"]

        self.n_avail_samples = len(self.HF_field)
        assert len(self.LF_field) == self.n_avail_samples
        assert len(self.HF_field_prediction_flora) == self.n_avail_samples
        assert len(self.HF_field_prediction_floral) == self.n_avail_samples

        # number of UQ samples
        self.n_gen = self.HF_field_prediction_floral.shape[1]
        assert self.n_gen == self.HF_field_prediction_flora.shape[1]

        # compute mean and std over the UQ samples (B, C, *dims)
        self.mean_dict = {
            "Low-fidelity": self.LF_field,
            "High-fidelity": self.HF_field,
            "FLORA": self.HF_field_prediction_flora.mean(1),
            "FLORAL": self.HF_field_prediction_floral.mean(1),
        }

        self.std_dict = {
            "Low-fidelity": torch.ones_like(self.LF_field) * 1e-6,
            "FLORA": self.HF_field_prediction_flora.std(1, correction=0),
            "FLORAL": self.HF_field_prediction_floral.std(1, correction=0),
        }
        self.error_dict = self._get_error_dict()

    def _get_error_dict(self):
        error_dict = {
            "abs_error": {},
            "rel_error": {},
            "RMSE": {},
            "NRMSE": {},
            "CRMSE": {},
        }
        for k in self.mean_dict:
            true = self.mean_dict["High-fidelity"]
            pred = self.mean_dict[k]
            if k != "High-fidelity":
                # absolute error
                abs_error = (pred - true).abs()
                error_dict["abs_error"][k] = abs_error
                # relative error
                rel_error = ((pred - true) / true).abs()
                error_dict["rel_error"][k] = rel_error
                # RMSE
                RMSE = compute_RMSE(true=true, pred=pred)
                error_dict["RMSE"][k] = RMSE
                # NRMSE
                NRMSE = compute_NRMSE(true=true, pred=pred)
                error_dict["NRMSE"][k] = NRMSE
                # CRMSE
                CRMSE = compute_CRMSE(true=true, pred=pred)
                error_dict["CRMSE"][k] = CRMSE

        print(
            f"\n=== Error summary for n_train: {self.n_train} "
            f"and n_val: {self.n_val} ==="
        )
        print_error_summary(error_dict)
        return error_dict

    def get_error_summary(self):
        print(
            f"\n=== Error summary for n_train: {self.n_train} "
            f"and n_val: {self.n_val} ==="
        )
        print_error_summary(self.error_dict)

    def make_field_sample_plot(
        self, n_samples: int = 5, plot_channel: int = 0, **kwargs
    ):
        """make field samples plot"""
        assert n_samples <= self.n_avail_samples
        print(f"Plotting {n_samples} samples for channel: {plot_channel}")
        independent_colorbar = kwargs.get("independent_colorbar", True)
        fig, axs = plt.subplots(
            n_samples,
            len(self.mean_dict),
            figsize=kwargs.get("figsize", (len(self.mean_dict) * 3.0, n_samples * 2.0)),
            dpi=300,
            layout="compressed",
            sharex=True,
            sharey=True,
        )

        if independent_colorbar:
            # compute the range for each row
            vmin_list = [
                float(min(data[ii].min() for data in self.mean_dict.values()))
                for ii in range(n_samples)
            ]
            vmax_list = [
                float(max(data[ii].max() for data in self.mean_dict.values()))
                for ii in range(n_samples)
            ]
        else:
            # compute range
            if kwargs.get("vmin", None) is None:
                vmin = min(data[:n_samples].min() for data in self.mean_dict.values())
                vmin = torch.floor(vmin)
            else:
                vmin = kwargs.get("vmin")
            if kwargs.get("vmax", None) is None:
                vmax = max(data[:n_samples].max() for data in self.mean_dict.values())
                vmax = torch.ceil(vmax)
            else:
                vmax = kwargs.get("vmax")

            vmin_list = [vmin] * n_samples
            vmax_list = [vmax] * n_samples

        for jj, k in enumerate(self.mean_dict.keys()):
            axs[0, jj].set_title(k)
            for ii in range(n_samples):
                im = axs[ii, jj].imshow(
                    self.mean_dict[k][ii][plot_channel],
                    vmin=vmin_list[ii],
                    vmax=vmax_list[ii],
                    origin="lower",
                    interpolation="bicubic",
                )

                if (independent_colorbar is True) and jj == 0:
                    fig.colorbar(im, ax=axs[ii, -1], orientation="vertical", pad=0.1)

        if independent_colorbar is False:
            fig.colorbar(im, ax=axs[-1], orientation="horizontal", pad=0.1)
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(kwargs.get("xlabel", r"$x_1$"))
            ax.set_ylabel(kwargs.get("ylabel", r"$x_2$"))
            ax.label_outer()
        default_save_name = f"field_samples_n_train_{self.n_train}_n_val_{self.n_val}"
        plt.savefig(kwargs.get("save_name", default_save_name) + ".png")

    def make_error_sample_plot(
        self, n_samples: int = 5, plot_channel: int = 0, **kwargs
    ):
        """make field samples plot"""
        assert n_samples <= self.n_avail_samples
        error_type = kwargs.get("error_type", "abs_error")
        assert error_type in ["abs_error", "rel_error"], "invalid error type"
        print(
            f"Plotting {n_samples} samples for channel: {plot_channel} "
            f"for error type: {error_type}"
        )
        independent_colorbar = kwargs.get("independent_colorbar", True)
        data_dict = self.error_dict[error_type]

        fig, axs = plt.subplots(
            n_samples,
            len(data_dict),
            figsize=kwargs.get("figsize", (len(data_dict) * 2.4, n_samples * 2.0)),
            dpi=300,
            layout="compressed",
            sharex=True,
            sharey=True,
        )
        if independent_colorbar:
            # compute the range for each row
            vmin_list = [
                float(min(data[ii].min() for data in data_dict.values()))
                for ii in range(n_samples)
            ]
            vmax_list = [
                float(max(data[ii].max() for data in data_dict.values()))
                for ii in range(n_samples)
            ]
        else:
            # compute range
            if kwargs.get("vmin", None) is None:
                vmin = min(data[:n_samples].min() for data in data_dict.values())
                vmin = torch.floor(vmin)
            else:
                vmin = kwargs.get("vmin")
            if kwargs.get("vmax", None) is None:
                vmax = max(data[:n_samples].max() for data in data_dict.values())
                vmax = torch.ceil(vmax)
            else:
                vmax = kwargs.get("vmax")

            vmin_list = [vmin] * n_samples
            vmax_list = [vmax] * n_samples

        for jj, k in enumerate(data_dict.keys()):
            axs[0, jj].set_title(k)
            for ii in range(n_samples):
                im = axs[ii, jj].imshow(
                    data_dict[k][ii][plot_channel],
                    vmin=vmin_list[ii],
                    vmax=vmax_list[ii],
                    origin="lower",
                    interpolation="bicubic",
                )

                if (independent_colorbar is True) and jj == 0:
                    fig.colorbar(im, ax=axs[ii, -1], orientation="vertical", pad=0.1)

        if independent_colorbar is False:
            fig.colorbar(im, ax=axs[-1], orientation="horizontal", pad=0.1)
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(kwargs.get("xlabel", r"$x_1$"))
            ax.set_ylabel(kwargs.get("ylabel", r"$x_2$"))
            ax.label_outer()
        default_save_name = (
            f"{error_type}_samples_n_train_{self.n_train}_n_val_{self.n_val}"
        )
        plt.savefig(kwargs.get("save_name", default_save_name) + ".png")


class oneDPlot(BasePlot):
    """Plot multiple samples"""

    def __init__(
        self,
        data_flora: dict,
        data_floral: dict,
        data_fno_flora: dict,
        data_fno_floral: dict,
    ):
        super(oneDPlot, self).__init__()
        self.data_flora = data_flora
        self.data_floral = data_floral

        self.data_fno_flora = data_fno_flora
        self.data_fno_floral = data_fno_floral

        # High-fidelity data (B, n_fields_per_sample, channels, *dims)
        self.HF_field = self.data_floral["HF_field"]
        # Low-fidelity data (B, n_fields_per_sample, channels, *dims)
        self.LF_field = self.data_floral["LF_field"]
        # Prediction Flora (B, n_fields_per_sample*n_gen, channels, *dims)
        self.HF_field_prediction_flora = self.data_flora["prediction"]
        self.HF_field_prediction_fno_flora = self.data_fno_flora["prediction"]
        # Prediction Floral (B, n_fields_per_sample*n_gen, channels, *dims)
        self.HF_field_prediction_floral = self.data_floral["prediction"]
        self.HF_field_prediction_fno_floral = self.data_fno_floral["prediction"]

        # extract num train and val
        self.n_train = self.data_floral["n_train"]  # total number of samples
        self.n_val = self.data_floral["n_val"]  # total number of samples
        self.n_test = self.data_floral["n_test"]  # total number of samples
        # domains
        self.field_domain = self.data_flora["domain_dict"]["field"].ravel()

        self.n_avail_samples = self.data_floral["n_unique_test"]
        assert len(self.LF_field) == self.n_avail_samples
        assert len(self.HF_field_prediction_flora) == self.n_avail_samples
        assert len(self.HF_field_prediction_floral) == self.n_avail_samples
        assert len(self.HF_field_prediction_fno_flora) == self.n_avail_samples
        assert len(self.HF_field_prediction_fno_floral) == self.n_avail_samples
        # number of UQ samples
        self.n_fields_per_sample = self.data_floral["n_fields_per_sample"]
        self.n_gen = (
            self.HF_field_prediction_floral.shape[1] // self.n_fields_per_sample
        )
        assert (
            self.HF_field_prediction_floral.shape[1]
            == self.n_fields_per_sample * self.n_gen
        )

        # compute mean and std over the UQ samples (B, C, *dims)
        self.mean_dict = {
            "Low-fidelity": self.LF_field.mean(1),
            "High-fidelity": self.HF_field.mean(1),
            "FLORA": self.HF_field_prediction_flora.mean(1),
            "FLORAL": self.HF_field_prediction_floral.mean(1),
            "FNO": self.HF_field_prediction_fno_flora.mean(1),
            "Residual FNO": self.HF_field_prediction_fno_floral.mean(1),
        }

        self.std_dict = {
            "Low-fidelity": self.LF_field.std(1, correction=0),
            "High-fidelity": self.HF_field.std(1, correction=0),
            "FLORA": self.HF_field_prediction_flora.std(1, correction=0),
            "FLORAL": self.HF_field_prediction_floral.std(1, correction=0),
            "FNO": self.HF_field_prediction_fno_flora.std(1, correction=0),
            "Residual FNO": self.HF_field_prediction_fno_floral.std(1, correction=0),
        }

        self.error_dict = self._get_error_dict()

    def _get_error_dict(self):
        error_dict = {
            "abs_error": {},
            "rel_error": {},
            "RMSE": {},
            "NRMSE": {},
            "CRMSE": {},
        }
        for k in self.mean_dict:
            true = self.mean_dict["High-fidelity"]
            pred = self.mean_dict[k]
            if k != "High-fidelity":
                # absolute error
                abs_error = (pred - true).abs()
                error_dict["abs_error"][k] = abs_error
                # relative error
                rel_error = ((pred - true) / true).abs()
                error_dict["rel_error"][k] = rel_error
                # RMSE
                RMSE = compute_RMSE(true=true, pred=pred)
                error_dict["RMSE"][k] = RMSE
                # NRMSE
                NRMSE = compute_NRMSE(true=true, pred=pred)
                error_dict["NRMSE"][k] = NRMSE
                # CRMSE
                CRMSE = compute_CRMSE(true=true, pred=pred)
                error_dict["CRMSE"][k] = CRMSE

        print(
            f"\n=== Error summary for n_train: {self.n_train} "
            f"and n_val: {self.n_val} ==="
        )
        print_error_summary(error_dict)
        return error_dict

    def get_error_summary(self):
        print(
            f"\n=== Error summary for n_train: {self.n_train} "
            f"and n_val: {self.n_val} ==="
        )
        print_error_summary(self.error_dict)

    def make_field_sample_plot(
        self, n_samples: int = 5, plot_channel: int = 0, **kwargs
    ):
        """make field samples plot — row 0: mean, row 1: variance"""
        assert n_samples <= self.n_avail_samples
        print(f"Plotting {n_samples} samples for channel: {plot_channel}")
        linewidth = kwargs.get("linewidth", 2.5)
        fig, axs = plt.subplots(
            2,
            n_samples,
            figsize=kwargs.get("figsize", (n_samples * 3.5, 6.0)),
            dpi=300,
            layout="compressed",
            sharex=True,
            sharey="row",
        )
        if n_samples == 1:
            axs = axs.reshape(2, 1)
        _plot_colors = _METHOD_COLORS
        _linestyles = {
            "High-fidelity": "-",
            "Low-fidelity": "--",
            "FLORA": "-",
            "FLORAL": "-.",
            "FNO": "-",
            "Residual FNO": "-.",
        }
        _mean_order = [
            "High-fidelity",
            "Low-fidelity",
            "FNO",
            "Residual FNO",
            "FLORA",
            "FLORAL",
        ]
        _var_order = ["High-fidelity", "FLORA", "FLORAL"]

        # Row 0: mean prediction
        for k in _mean_order:
            if k not in self.mean_dict:
                continue
            line_kwargs = dict(
                color=_plot_colors[k], linestyle=_linestyles[k], linewidth=linewidth
            )
            for ii in range(n_samples):
                mean_pred = self.mean_dict[k][ii][plot_channel].ravel()
                axs[0, ii].plot(self.field_domain, mean_pred, label=k, **line_kwargs)

        # Row 1: variance (std^2) — deterministic models excluded
        for k in _var_order:
            if k not in self.std_dict:
                continue
            line_kwargs = dict(
                color=_plot_colors[k], linestyle=_linestyles[k], linewidth=linewidth
            )
            for ii in range(n_samples):
                var_pred = self.std_dict[k][ii][plot_channel].ravel() ** 2
                axs[1, ii].plot(self.field_domain, var_pred, label=k, **line_kwargs)
                axs[1, ii].set_yscale("log")

        # inset on top-right subplot (only when inset_xlim is provided)
        inset_xlim = kwargs.get("inset_xlim", None)
        inset_ylim = kwargs.get("inset_ylim", None)
        inset_bounds = kwargs.get("inset_bounds", [0.55, 0.05, 0.42, 0.42])
        if inset_xlim is not None:
            ax_top_right = axs[0, -1]
            ax_inset = ax_top_right.inset_axes(inset_bounds)
            for k in _mean_order:
                if k not in self.mean_dict:
                    continue
                line_kwargs = dict(
                    color=_plot_colors[k],
                    linestyle=_linestyles[k],
                    linewidth=linewidth * 0.8,
                )
                mean_pred = self.mean_dict[k][-1][plot_channel].ravel()
                ax_inset.plot(self.field_domain, mean_pred, **line_kwargs)
            ax_inset.set_xlim(inset_xlim)
            if inset_ylim is not None:
                ax_inset.set_ylim(inset_ylim)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            for spine in ax_inset.spines.values():
                spine.set_linewidth(0.8)
            # box in main axes + connector lines to inset
            ax_top_right.indicate_inset_zoom(ax_inset, edgecolor="black", linewidth=0.8)

        label_fontsize = kwargs.get("label_fontsize", 14)
        xlabel = kwargs.get("xlabel", r"$x$")
        # x label only on bottom row; label_outer hides inner tick labels
        for ii in range(n_samples):
            axs[1, ii].set_xlabel(xlabel, fontsize=label_fontsize)
        axs[0, 0].set_ylabel(
            kwargs.get("ylabel", r"$\mathbb{E}[w(x)]$"), fontsize=label_fontsize
        )
        axs[1, 0].set_ylabel(r"$\mathrm{Var}[w(x)]$", fontsize=label_fontsize)
        for ax in axs.flat:
            ax.label_outer()

        # figure-level legend centered vertically between the two rows
        handles, labels = axs[0, 0].get_legend_handles_labels()
        leg = fig.legend(
            handles,
            labels,
            framealpha=1.0,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            bbox_transform=fig.transFigure,
            borderaxespad=0.0,
        )
        leg.set_zorder(100)
        fig.align_ylabels()

        default_save_name = f"field_samples_n_train_{self.n_train}_n_val_{self.n_val}"
        plt.savefig(
            kwargs.get("save_name", default_save_name) + ".png",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def make_error_sample_plot(
        self, n_samples: int = 5, plot_channel: int = 0, **kwargs
    ):
        """make field samples plot"""
        raise NotImplementedError


class ParetoPlot:
    """Pareto plot class"""

    @classmethod
    def get_pareto_data(
        self, data_flora, data_floral, model: str = None, deterministic: bool = False
    ):
        """extract the pareto data"""
        # High-fidelity data (B, n_fields_per_sample, channels, *dims)
        HF_field = data_floral["HF_field"]
        # Low-fidelity data (B, n_fields_per_sample, channels, *dims)
        LF_field = data_floral["LF_field"]
        # Prediction Flora
        HF_field_prediction_flora = data_flora["prediction"]
        # Prediction Floral
        HF_field_prediction_floral = data_floral["prediction"]

        n_avail_samples = data_floral.get("n_unique_test")
        assert len(LF_field) == n_avail_samples
        assert len(HF_field_prediction_flora) == n_avail_samples
        assert len(HF_field_prediction_floral) == n_avail_samples

        n_fields_per_sample = data_floral["n_fields_per_sample"]
        n_gen = HF_field_prediction_floral.shape[1] // n_fields_per_sample
        assert HF_field_prediction_floral.shape[1] == n_fields_per_sample * n_gen

        # mean field
        mean_dict = {
            "Low-fidelity": LF_field.mean(1),
            "High-fidelity": HF_field.mean(1),
            "FLORA": HF_field_prediction_flora.mean(1),
            "FLORAL": HF_field_prediction_floral.mean(1),
        }

        # variance field
        variance_dict = {
            "Low-fidelity": LF_field.var(1),
            "High-fidelity": HF_field.var(1),
            "FLORA": HF_field_prediction_flora.var(1),
            "FLORAL": HF_field_prediction_floral.var(1),
        }

        # mean field error
        true_mean_field = mean_dict.get("High-fidelity")
        mean_field_error = {
            k: torch.norm(
                (mean_dict[k] - true_mean_field).flatten(1),
                dim=1,
            )
            for k in mean_dict
            if k != "High-fidelity"
        }
        # variance field error
        true_variance_field = variance_dict.get("High-fidelity")
        variance_field_error = {
            k: torch.norm(
                (variance_dict[k] - true_variance_field).flatten(1),
                dim=1,
            )
            for k in variance_dict
            if k != "High-fidelity"
        }

        # Convert to rows for each method and unique condition
        rows = []
        for method in mean_field_error.keys():
            for mean_err, var_err in zip(
                mean_field_error[method].tolist(),
                variance_field_error[method].tolist(),
            ):
                rows.append(
                    {
                        "Model": model,
                        "Method": method,
                        "Mean Field Error": mean_err,
                        "Variance Field Error": var_err,
                    }
                )

        summary_df = pd.DataFrame(rows)

        method_order = ["Low-fidelity", "FLORA", "FLORAL"]
        cat_type = pd.CategoricalDtype(categories=method_order, ordered=True)
        summary_df["Method"] = summary_df["Method"].astype(cat_type)
        summary_df = (
            summary_df.groupby(["Model", "Method"], observed=False)[
                ["Mean Field Error", "Variance Field Error"]
            ]
            .mean()
            .reset_index()
        )

        summary_df["Samples (total)"] = (
            data_flora["n_train"] + data_flora["n_val"] + data_flora["n_test"]
        )
        summary_df["Samples (train)"] = data_flora["n_train"]
        summary_df["Samples (validation)"] = data_flora["n_val"]

        return summary_df

    @classmethod
    def plot_pareto_res(self, combined_df, **kwargs):
        def _make_label(r):
            if r["Model"] == "FNO":
                return "FNO" if r["Method"] == "FLORA" else "Residual FNO"
            return r["Method"]

        # Split data
        df_lf = combined_df[combined_df["Method"] == "Low-fidelity"].iloc[[0]]
        df_rest = combined_df[combined_df["Method"] != "Low-fidelity"].copy()
        df_rest["Label"] = df_rest.apply(_make_label, axis=1)
        label_order = df_rest["Label"].unique().tolist()
        palette = {k: _METHOD_COLORS[k] for k in label_order if k in _METHOD_COLORS}

        figsize = kwargs.get("figsize", (10, 5))
        plt.figure(figsize=figsize, layout="compressed")
        ax = sns.scatterplot(
            data=df_lf,
            x="Variance Field Error",
            y="Mean Field Error",
            color="#999999",
            s=200,
            marker="*",
            edgecolor="black",
            label="Low-fidelity (reference)",
        )

        sns.scatterplot(
            data=df_rest,
            x="Variance Field Error",
            y="Mean Field Error",
            hue="Label",
            hue_order=label_order,
            style="Resolution (train)",
            palette=palette,
            s=100,
            edgecolor="black",
            ax=ax,
        )

        ax.set_yscale("log")
        ax.set_xscale("log")
        xlim_range = kwargs.get("xlim_range", (1e-7, 1e-1))
        ylim_range = kwargs.get("ylim_range", (1e-1, 1e2))
        ax.set_xlim(left=xlim_range[0], right=xlim_range[1])
        ax.set_ylim(bottom=ylim_range[0], top=ylim_range[1])

        # shade region where deterministic models (FNO, Residual FNO) live
        det_threshold = kwargs.get("det_threshold", 1e-5)
        ax.axvspan(xlim_range[0], det_threshold, color="gray", alpha=0.1, zorder=0)
        ax.axvline(det_threshold, color="gray", linestyle=":", linewidth=1.5, zorder=1)
        ax.text(
            det_threshold * 0.5,
            ylim_range[1] * 0.6,
            "Deterministic",
            fontsize=15,
            color="gray",
            ha="center",
            va="top",
            rotation=90,
        )

        handles, labels = ax.get_legend_handles_labels()
        default_legend_kwargs = {
            "bbox_to_anchor": (1.01, 1),
            "loc": "upper left",
            "borderaxespad": 0.0,
        }
        ax.legend(
            handles,
            labels,
            title="",
            **kwargs.get("legend_kwargs", default_legend_kwargs),
        )
        ax.set_xlabel(r"Variance Field Error $L_2$ norm")
        ax.set_ylabel(r"Mean Field Error $L_2$ norm")
        plt.savefig(
            "pareto_resolution_comparison.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()

    @classmethod
    def plot_pareto(self, combined_df, det_as_lines: bool = True, **kwargs):
        """Plot the Pareto front.

        Parameters
        ----------
        combined_df : pd.DataFrame
            Output of repeated calls to ``get_pareto_data``, concatenated.
        det_as_lines : bool
            If True (default) deterministic models (FNO, Residual FNO) are drawn
            as horizontal lines with linestyle encoding Samples (train).
            If False they are drawn as scatter markers just like the probabilistic
            models, using their actual (Mean Std, L2 Error) position.
        """
        from matplotlib.lines import Line2D
        from matplotlib.legend_handler import HandlerTuple

        def _make_label(r):
            if r["Model"] == "FNO":
                return "FNO" if r["Method"] == "FLORA" else "Residual FNO"
            return r["Method"]

        # Low-fidelity: single reference point (take one row)
        df_lf = combined_df[combined_df["Method"] == "Low-fidelity"].iloc[[0]]
        df_rest = combined_df[combined_df["Method"] != "Low-fidelity"].copy()
        df_rest["Label"] = df_rest.apply(_make_label, axis=1)

        prob_labels = ["FLORA", "FLORAL"]
        det_labels = ["FNO", "Residual FNO"]
        df_prob = df_rest[df_rest["Label"].isin(prob_labels)].copy()
        df_det = df_rest[df_rest["Label"].isin(det_labels)].copy()

        palette = {
            k: _METHOD_COLORS[k]
            for k in prob_labels + det_labels
            if k in _METHOD_COLORS
        }

        # marker / linestyle per training-sample count (uniform linewidth)
        n_train_vals = sorted(combined_df["Samples (train)"].unique())
        _markers = ["o", "s", "D", "^", "v"]
        # explicit dash patterns — clearly distinct at all scales
        _linestyles = [
            (0, ()),  # solid
            (0, (7, 2)),  # long dash
            (0, (2, 2)),  # short dash
            (0, (7, 2, 2, 2)),  # dash-dot
            (0, (1, 1)),  # densely dotted
        ]
        _lw = 1.5  # uniform linewidth for all deterministic lines
        train_to_marker = {
            n: _markers[i % len(_markers)] for i, n in enumerate(n_train_vals)
        }
        train_to_ls = {
            n: _linestyles[i % len(_linestyles)] for i, n in enumerate(n_train_vals)
        }

        figsize = kwargs.get("figsize", (10, 5))
        xlim_range = kwargs.get("xlim_range", (1e-7, 1e-1))
        ylim_range = kwargs.get("ylim_range", (1e-1, 1e2))

        fig, ax = plt.subplots(1, 1, figsize=figsize, layout="compressed")

        # Low-fidelity reference
        ax.scatter(
            df_lf["Variance Field Error"],
            df_lf["Mean Field Error"],
            color="#999999",
            s=200,
            marker="*",
            edgecolor="black",
            zorder=5,
        )

        # Probabilistic: scatter markers, one per (model, n_train)
        for label in prob_labels:
            color = palette[label]
            for n_train in n_train_vals:
                sub = df_prob[
                    (df_prob["Label"] == label)
                    & (df_prob["Samples (train)"] == n_train)
                ]
                if sub.empty:
                    continue
                ax.scatter(
                    sub["Variance Field Error"],
                    sub["Mean Field Error"],
                    color=color,
                    s=100,
                    marker=train_to_marker[n_train],
                    edgecolor="black",
                    zorder=5,
                )

        # Deterministic: horizontal lines or scatter depending on flag
        for label in det_labels:
            color = palette[label]
            for n_train in n_train_vals:
                sub = df_det[
                    (df_det["Label"] == label) & (df_det["Samples (train)"] == n_train)
                ]
                if sub.empty:
                    continue
                if det_as_lines:
                    ax.scatter(
                        sub["Variance Field Error"],
                        sub["Mean Field Error"],
                        color=color,
                        s=100,
                        marker=train_to_marker[n_train],
                        edgecolor="black",
                        zorder=5,
                    )
                else:
                    ax.scatter(
                        sub["Variance Field Error"],
                        sub["Mean Field Error"],
                        color=color,
                        s=100,
                        marker=train_to_marker[n_train],
                        edgecolor="black",
                        zorder=5,
                    )

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(left=xlim_range[0], right=xlim_range[1])
        ax.set_ylim(bottom=ylim_range[0], top=ylim_range[1])

        # ---- Legend 1: model colors ----
        # deterministic legend entries switch between line and marker icons
        if det_as_lines:
            det_model_handles = [
                Line2D([0], [0], color=palette["FNO"], linewidth=2, label="FNO"),
                Line2D(
                    [0],
                    [0],
                    color=palette["Residual FNO"],
                    linewidth=2,
                    label="Residual FNO",
                ),
            ]
        else:
            det_model_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=palette["FNO"],
                    markersize=9,
                    markeredgecolor="black",
                    label="FNO",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=palette["Residual FNO"],
                    markersize=9,
                    markeredgecolor="black",
                    label="Residual FNO",
                ),
            ]

        model_handles = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="#999999",
                markersize=12,
                markeredgecolor="black",
                label="Low-fidelity",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=palette["FLORA"],
                markersize=9,
                markeredgecolor="black",
                label="FLORA",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=palette["FLORAL"],
                markersize=9,
                markeredgecolor="black",
                label="FLORAL",
            ),
            *det_model_handles,
        ]

        # model legend anchored to the top of the axes
        leg1 = ax.legend(
            handles=model_handles,
            labels=[h.get_label() for h in model_handles],
            title="Model",
            bbox_to_anchor=kwargs.get("model_legend_anchor", (1.01, 1.0)),
            loc="upper left",
            borderaxespad=0.0,
        )
        ax.add_artist(leg1)

        # ---- Legend 2: Samples (train) ----
        # det_as_lines=True  → two columns per row: (marker, line)
        # det_as_lines=False → single marker column
        samples_handles, samples_labels_list = [], []
        for n_train in n_train_vals:
            marker_h = Line2D(
                [0],
                [0],
                marker=train_to_marker[n_train],
                color="w",
                markerfacecolor="gray",
                markersize=8,
                markeredgecolor="black",
            )
            if det_as_lines:
                line_h = Line2D(
                    [0],
                    [0],
                    color="gray",
                    linestyle=train_to_ls[n_train],
                    linewidth=_lw,
                )
                samples_handles.append((marker_h, line_h))
            else:
                samples_handles.append(marker_h)
            samples_labels_list.append(str(n_train))

        # anchor samples legend to the bottom so it never overlaps the model legend
        samples_legend_kwargs = dict(
            handles=samples_handles,
            labels=samples_labels_list,
            title="Samples (train)",
            handletextpad=0.8,
            bbox_to_anchor=kwargs.get("samples_legend_anchor", (1.01, 0.0)),
            loc="lower left",
            borderaxespad=0.0,
        )
        if det_as_lines:
            samples_legend_kwargs["handler_map"] = {
                tuple: HandlerTuple(ndivide=None, pad=0.5)
            }
            samples_legend_kwargs["handlelength"] = 4.0
        ax.legend(**samples_legend_kwargs)

        ax.set_xlabel(r"Variance Field Error $L_2$ norm")
        ax.set_ylabel(r"Mean Field Error $L_2$ norm")
        plt.savefig(
            "pareto_comparison.png", dpi=300, bbox_inches="tight", pad_inches=0.1
        )
        plt.close()
