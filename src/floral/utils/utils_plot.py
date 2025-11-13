"""
Plotting utilities
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from abc import abstractmethod, ABC


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


class ErrorSummary:
    def __init__(self, data_flora: dict, data_floral: dict):
        super(ErrorSummary, self).__init__()
        self.data_flora = data_flora
        self.data_floral = data_floral

        # High-fidelity data (B, channels, *dims)
        self.HF_field = self.data_floral["HF_field_plot"]
        # Low-fidelity data (B, channels, *dims)
        self.LF_field = self.data_floral["LF_field_plot"]
        # Prediction Flora
        self.HF_field_prediction_flora = self.data_flora["HF_field_prediction_plot"]
        # Prediction Floral
        self.HF_field_prediction_floral = self.data_floral["HF_field_prediction_plot"]
        # extract num train and val
        self.n_train = self.data_floral["n_train"]
        self.n_val = self.data_floral["n_val"]
        self.n_samples = self.data_floral["n_samples"]

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

    def __call__(self, verbose: bool = False):
        error_dict = {
            # "abs_error": {},
            # "rel_error": {},
            "RMSE": {},
            "NRMSE": {},
            "CRMSE": {},
        }

        for k in self.mean_dict:
            true = self.mean_dict["High-fidelity"]
            pred = self.mean_dict[k]
            if k != "High-fidelity":
                # # absolute error
                # abs_error = (pred - true).abs()
                # error_dict["abs_error"][k] = abs_error
                # # relative error
                # rel_error = ((pred - true) / true).abs()
                # error_dict["rel_error"][k] = rel_error
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
                        "Method": fidelity,
                        "Metric": metric,
                        "Value": val.item() if hasattr(val, "item") else val,
                    }
                )
        return pd.DataFrame(df)

    @staticmethod
    def plot_error(combined_df, **kwargs):
        """plot the errors"""
        metrics = ["RMSE", "NRMSE", "CRMSE"]
        # Create 1×3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="compressed")
        for ii, (ax, metric) in enumerate(zip(axes, metrics)):
            # Filter dataframe for each metric
            subset = combined_df[combined_df["Metric"] == metric]

            # Scatter plot
            sns.scatterplot(
                data=subset,
                x="Samples (train)",
                y="Value",
                style="Method",
                style_order=["Low-fidelity", "FLORA", "FLORAL"],
                hue="Method",
                hue_order=["Low-fidelity", "FLORA", "FLORAL"],
                palette="Set2",
                s=150,
                edgecolor="black",
                ax=ax,
            )

            # Titles and labels
            # ax.set_title(metric)
            ax.set_xlabel("Samples (train)", fontsize=15)
            ax.set_ylabel(metric, fontsize=15)
            if ii == 0:
                ax.legend().set_title("Method")
            else:
                ax.legend_.remove()
            ax.set_yscale("log")
            ax.set_xscale("log")
            xlim_range = kwargs.get("xlim_range", (1e1, 1e4))
            ylim_range = kwargs.get("ylim_range", (1e-2, 1e1))
            ax.set_xlim(left=xlim_range[0], right=xlim_range[1])
            ax.set_ylim(bottom=ylim_range[0], top=ylim_range[1])

        plt.savefig("error_comparison.png", dpi=300)
        plt.close()


class twoDPlot(BasePlot):
    """Plot multiple samples"""

    def __init__(self, data_flora: dict, data_floral: dict):
        super(twoDPlot, self).__init__()
        self.data_flora = data_flora
        self.data_floral = data_floral

        # High-fidelity data (B, channels, *dims)
        self.HF_field = self.data_floral["HF_field_plot"]
        # Low-fidelity data (B, channels, *dims)
        self.LF_field = self.data_floral["LF_field_plot"]
        # Prediction Flora
        self.HF_field_prediction_flora = self.data_flora["HF_field_prediction_plot"]
        # Prediction Floral
        self.HF_field_prediction_floral = self.data_floral["HF_field_prediction_plot"]
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
            "FLORA": self.HF_field_prediction_flora.std(1),
            "FLORAL": self.HF_field_prediction_floral.std(1),
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
        fig, axs = plt.subplots(
            n_samples,
            len(self.mean_dict),
            figsize=kwargs.get("figsize", (len(self.mean_dict) * 3.0, n_samples * 2.0)),
            dpi=300,
            layout="compressed",
            sharex=True,
            sharey=True,
        )
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
        for jj, k in enumerate(self.mean_dict.keys()):
            axs[0, jj].set_title(k)
            for ii in range(n_samples):
                im = axs[ii, jj].imshow(
                    self.mean_dict[k][ii][plot_channel],
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                    interpolation="bicubic",
                )
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
        for jj, k in enumerate(data_dict.keys()):
            axs[0, jj].set_title(k)
            for ii in range(n_samples):
                im = axs[ii, jj].imshow(
                    data_dict[k][ii][plot_channel],
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                    interpolation="bicubic",
                )
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

    def __init__(self, data_flora: dict, data_floral: dict):
        super(oneDPlot, self).__init__()
        self.data_flora = data_flora
        self.data_floral = data_floral

        # High-fidelity data (B, channels, *dims)
        self.HF_field = self.data_floral["HF_field_plot"]
        # Low-fidelity data (B, channels, *dims)
        self.LF_field = self.data_floral["LF_field_plot"]
        # Prediction Flora
        self.HF_field_prediction_flora = self.data_flora["HF_field_prediction_plot"]
        # Prediction Floral
        self.HF_field_prediction_floral = self.data_floral["HF_field_prediction_plot"]
        # extract num train and val
        self.n_train = self.data_floral["n_train"]
        self.n_val = self.data_floral["n_val"]
        # domains
        self.field_domain = self.data_flora["domain_dict"]["field"].ravel()

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
            "FLORA": self.HF_field_prediction_flora.std(1),
            "FLORAL": self.HF_field_prediction_floral.std(1),
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
        std_factor = kwargs.get("std_factor", 1.0)
        print(f"Plotting {std_factor} standard deviation")
        assert n_samples <= self.n_avail_samples
        print(f"Plotting {n_samples} samples for channel: {plot_channel}")
        fig, axs = plt.subplots(
            n_samples,
            2,
            figsize=kwargs.get("figsize", (len(self.mean_dict) * 3.0, n_samples * 2.0)),
            dpi=300,
            layout="compressed",
            sharex=True,
            # sharey=True,
        )
        for k in self.mean_dict.keys():
            if k == "High-fidelity":
                line_kwargs = dict(color="black", linestyle="-", linewidth=2)
            elif k == "Low-fidelity":
                line_kwargs = dict(color="grey", linestyle="--", linewidth=2)
            elif k == "FLORA":
                line_kwargs = dict(color="red", linestyle="-", linewidth=2)
            elif k == "FLORAL":
                line_kwargs = dict(color="blue", linestyle="-.", linewidth=2)
            else:
                raise ValueError(f"{k} not a valid entry")

            for ii in range(n_samples):
                mean_pred = self.mean_dict[k][ii][plot_channel].ravel()
                (line_obj,) = axs[ii, 0].plot(
                    self.field_domain,
                    mean_pred,
                    label=k,
                    **line_kwargs,
                )

                if k in ["FLORA", "FLORAL"]:
                    std_pred = self.std_dict[k][ii][plot_channel].ravel()
                    axs[ii, 0].fill_between(
                        self.field_domain,
                        mean_pred + std_factor * std_pred,
                        mean_pred - std_factor * std_pred,
                        color=line_obj.get_color(),
                        alpha=0.25,
                        # label=f"$\pm{std_factor}\sigma$"
                        label=None,
                    )
                if k != "High-fidelity":
                    axs[ii, 1].plot(
                        self.field_domain,
                        torch.abs(
                            mean_pred
                            - self.mean_dict["High-fidelity"][ii][plot_channel].ravel()
                        ),
                        label=k,
                        **line_kwargs,
                    )
        for ii, ax in enumerate(axs.flatten()):
            if ii == 0:
                ax.legend()
            row = ii // 2
            # col = ii % 2
            # ax.set_xticks([])
            # ax.set_yticks([])
            if ii % 2 == 0:
                ax.set_ylabel(kwargs.get("ylabel", r"$w(x)$"))
            else:
                ax.set_ylabel(kwargs.get("xlabel", r"$|w(x) - \hat{w}(x)|$"))
            if row == n_samples - 1:
                ax.set_xlabel(kwargs.get("xlabel", r"$x$"))
            else:
                ax.set_xlabel("")  # remove label entirely
                ax.tick_params(labelbottom=False)  # hide tick labels
            # ax.label_outer()
        fig.align_ylabels()

        default_save_name = f"field_samples_n_train_{self.n_train}_n_val_{self.n_val}"
        plt.savefig(kwargs.get("save_name", default_save_name) + ".png")

    def make_error_sample_plot(
        self, n_samples: int = 5, plot_channel: int = 0, **kwargs
    ):
        """make field samples plot"""
        raise NotImplementedError

    #     assert n_samples <= self.n_avail_samples
    #     error_type = kwargs.get("error_type", "abs_error")
    #     assert error_type in ["abs_error", "rel_error"], "invalid error type"
    #     print(
    #         f"Plotting {n_samples} samples for channel: {plot_channel} "
    #         f"for error type: {error_type}"
    #     )
    #     data_dict = self.error_dict[error_type]

    #     fig, axs = plt.subplots(
    #         n_samples,
    #         len(data_dict),
    #         figsize=kwargs.get("figsize", (len(data_dict) * 2.4, n_samples * 2.0)),
    #         dpi=300,
    #         layout="compressed",
    #         sharex=True,
    #         sharey=True,
    #     )
    #     # compute range
    #     if kwargs.get("vmin", None) is None:
    #         vmin = min(data[:n_samples].min() for data in data_dict.values())
    #         vmin = torch.floor(vmin)
    #     else:
    #         vmin = kwargs.get("vmin")
    #     if kwargs.get("vmax", None) is None:
    #         vmax = max(data[:n_samples].max() for data in data_dict.values())
    #         vmax = torch.ceil(vmax)
    #     else:
    #         vmax = kwargs.get("vmax")
    #     for jj, k in enumerate(data_dict.keys()):
    #         axs[0, jj].set_title(k)
    #         for ii in range(n_samples):
    #             im = axs[ii, jj].imshow(
    #                 data_dict[k][ii][plot_channel],
    #                 vmin=vmin,
    #                 vmax=vmax,
    #                 origin="lower",
    #                 interpolation="bicubic",
    #             )
    #     fig.colorbar(im, ax=axs[-1], orientation="horizontal", pad=0.1)
    #     for ax in axs.flatten():
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.set_xlabel(kwargs.get("xlabel", r"$x_1$"))
    #         ax.set_ylabel(kwargs.get("ylabel", r"$x_2$"))
    #         ax.label_outer()
    #     default_save_name = (
    #         f"{error_type}_samples_n_train_{self.n_train}_n_val_{self.n_val}"
    #     )
    #     plt.savefig(kwargs.get("save_name", default_save_name) + ".png")


class ParetoPlot:
    """Pareto plot class"""

    @classmethod
    def get_pareto_data(self, data_flora, data_floral):
        """extract the pareto data"""
        # High-fidelity data (B, channels, *dims)
        HF_field = data_floral["HF_field_plot"]
        # Low-fidelity data (B, channels, *dims)
        LF_field = data_floral["LF_field_plot"]
        # Prediction Flora
        HF_field_prediction_flora = data_flora["HF_field_prediction_plot"]
        # Prediction Floral
        HF_field_prediction_floral = data_floral["HF_field_prediction_plot"]
        # extract num train and val
        # n_train = data_floral["n_train"]
        # n_val = data_floral["n_val"]

        n_avail_samples = len(HF_field)
        assert len(LF_field) == n_avail_samples
        assert len(HF_field_prediction_flora) == n_avail_samples
        assert len(HF_field_prediction_floral) == n_avail_samples

        # number of UQ samples
        n_gen = HF_field_prediction_floral.shape[1]
        assert n_gen == HF_field_prediction_flora.shape[1]

        mean_dict = {
            "Low-fidelity": LF_field,
            "FLORA": HF_field_prediction_flora.mean(1),
            "FLORAL": HF_field_prediction_floral.mean(1),
        }

        std_dict = {
            "Low-fidelity": torch.ones_like(LF_field) * 1e-6,
            "FLORA": HF_field_prediction_flora.std(1),
            "FLORAL": HF_field_prediction_floral.std(1),
        }

        # L2 norm
        l2_error_dict = {
            "Low-fidelity": torch.norm(
                HF_field.flatten(1) - LF_field.flatten(1), dim=1
            ),
            "FLORA": torch.norm(
                HF_field.flatten(1) - mean_dict["FLORA"].flatten(1), dim=1
            ),
            "FLORAL": torch.norm(
                HF_field.flatten(1) - mean_dict["FLORAL"].flatten(1), dim=1
            ),
        }

        # mean uncertainity (average uncertainity over the spatial domain)
        uncertainty_dict = {
            "Low-fidelity": torch.ones_like(l2_error_dict["Low-fidelity"])
            * 1e-6,  # no uncertainty for LF
            "FLORA": std_dict["FLORA"].flatten(1).mean(dim=1),
            "FLORAL": std_dict["FLORAL"].flatten(1).mean(dim=1),
        }

        # Convert to rows for each method
        error_rows, uncertainty_rows = [], []
        for method in mean_dict.keys():
            for e, u in zip(
                l2_error_dict[method].tolist(), uncertainty_dict[method].tolist()
            ):
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

        summary_df["Samples (total)"] = data_flora["n_samples"]
        summary_df["Samples (train)"] = data_flora["n_train"]
        summary_df["Samples (validation)"] = data_flora["n_val"]

        return summary_df

    @classmethod
    def plot_pareto(self, combined_df, **kwargs):
        # Split data
        df_lf = combined_df[combined_df["Method"] == "Low-fidelity"].iloc[
            [0]
        ]  # take one row
        df_rest = combined_df[combined_df["Method"] != "Low-fidelity"]

        # Plot methods that depend on Samples (Train)
        plt.figure(figsize=(10, 5), layout="compressed")
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
        xlim_range = kwargs.get("xlim_range", (1e-7, 1e-1))
        ylim_range = kwargs.get("ylim_range", (1e-1, 1e2))
        ax.set_xlim(left=xlim_range[0], right=xlim_range[1])
        ax.set_ylim(bottom=ylim_range[0], top=ylim_range[1])

        # Optional: make legend cleaner
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_xlabel("Mean Predictive Uncertainty (Mean Std)")
        ax.set_ylabel("Mean Predictive Error (L2 Norm)")
        plt.savefig("pareto_comparison.png", dpi=300)
        plt.close()
