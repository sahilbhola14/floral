# scripts/onedcorr/plot.py
"""
OneDCorr flow plots
"""
import os.path as osp
import pandas as pd
import torch
import matplotlib.pyplot as plt
from floral.utils import oneDPlot, ParetoPlot, ErrorSummary, BaseResidual

# Begin user input
n_train_samples_list = [100, 200, 500, 2000]
n_val_samples = 750
n_test_samples = 750
operator_method = "filmfno"
train_res = "Full"
results_folder = "./results"
# results_folder = "./dummy_res"
# End user input

plt.style.use("../journal.mplstyle")


def combine_path(paths: list):
    return osp.join(*paths)


def load_data(n_train_samples):
    nt = n_train_samples
    nv = n_val_samples
    ntest = n_test_samples
    method = operator_method.lower().strip()
    assert isinstance(train_res, (str, int))
    res = train_res.strip().lower() if isinstance(train_res, str) else train_res
    # file(s)
    file_flora = (
        f"onedcorr_fm_operator_method_{method}_n_train_{nt}_n_val_{nv}"
        f"_n_test_{ntest}_train_res_{res}_results_flora.pt"
    )

    file_floral = (
        f"onedcorr_fm_operator_method_{method}_n_train_{nt}_n_val_{nv}"
        f"_n_test_{ntest}_train_res_{res}_results_floral.pt"
    )

    file_fno_flora = (
        f"onedcorr_fno_operator_method_{method}_n_train_{nt}_n_val_{nv}"
        f"_n_test_{ntest}_train_res_{res}_results_flora.pt"
    )

    file_fno_floral = (
        f"onedcorr_fno_operator_method_{method}_n_train_{nt}_n_val_{nv}"
        f"_n_test_{ntest}_train_res_{res}_results_floral.pt"
    )

    print("flora file: ", file_flora)
    print("floral file: ", file_floral)

    print("(FNO) flora file: ", file_fno_flora)
    print("(FNO) floral file: ", file_fno_floral)

    print("n_train_samples: ", n_train_samples)
    print("n_val_samples: ", n_val_samples)

    data_flora = torch.load(
        combine_path([results_folder, file_flora]), weights_only=False
    )
    data_floral = torch.load(
        combine_path([results_folder, file_floral]), weights_only=False
    )

    data_fno_flora = torch.load(
        combine_path([results_folder, file_fno_flora]), weights_only=False
    )
    data_fno_floral = torch.load(
        combine_path([results_folder, file_fno_floral]), weights_only=False
    )

    print("Number of samples for UQ (Flora): " f"{data_flora['prediction'].shape[1]}")
    print("Number of samples for UQ (Floral): " f"{data_floral['prediction'].shape[1]}")

    print(
        "Number of samples for UQ (Flora): " f"{data_fno_flora['prediction'].shape[1]}"
    )

    print(
        "Number of samples for UQ (Floral): "
        f"{data_fno_floral['prediction'].shape[1]}"
    )

    def _print_cost(label, data, deterministic=False):
        t = data.get("inference_time_s")
        tpc = data.get("time_per_condition_s")
        tpcpg = data.get("time_per_condition_per_gen_s")
        nfe = data.get("nfe_per_condition")
        mem = data.get("peak_gpu_memory_mb")
        msg = f"[{label}] inference_time: {t:.2f} s | " f"time/condition: {tpc:.4f} s"
        if not deterministic:
            msg += f" | time/condition/gen: {tpcpg:.4f} s"
        if nfe is not None:
            msg += f" | NFE/condition: {nfe:.1f}"
        if mem is not None:
            msg += f" | peak GPU mem: {mem:.1f} MB"
        print(msg)

    _print_cost("FM  Flora ", data_flora)
    _print_cost("FM  Floral", data_floral)
    _print_cost("FNO Flora ", data_fno_flora, deterministic=True)
    _print_cost("FNO Floral", data_fno_floral, deterministic=True)

    print("--" * 10)

    return data_flora, data_floral, data_fno_flora, data_fno_floral


class ResidualOneDCorr(BaseResidual):
    def __init__(self, data_flora, data_floral):
        super(ResidualOneDCorr, self).__init__(
            data_flora=data_flora, data_floral=data_floral
        )
        # condition
        self.condition = self.full_condition[
            :, 0
        ]  # for onedcorr, only the first channel is the condition

    def comp_residual(self, prediction, condition, domain):
        """compute the residual for the oneDcorr equation"""
        true = self.condition.sin()
        return prediction - true


def plot_field(n_train_samples):
    # load the data
    data_flora, data_floral, data_fno_flora, data_fno_floral = load_data(
        n_train_samples
    )
    # create plot object
    plotter = oneDPlot(
        data_flora=data_flora,
        data_floral=data_floral,
        data_fno_flora=data_fno_flora,
        data_fno_floral=data_fno_floral,
    )
    # create sample plot
    plotter.make_field_sample_plot(
        n_samples=3,
        figsize=(12, 6),
        linewidth=2.5,
        label_fontsize=20,
        inset_xlim=(0.7, 1.0),
        inset_ylim=(-2, -1),
        inset_bounds=[0.55, 0.55, 0.42, 0.42],
    )
    # single-sample uncertainty band plot
    plotter.make_uq_band_plot(
        sample_idx=3,
        figsize=(12, 4),
        linewidth=2.5,
        label_fontsize=20,
        alpha=0.15,
        std_factor=5,
        inset_xlim=(0.75, 1.0),
        inset_ylim=(-1, 1),
    )


def print_pareto(n_train_samples_list):
    all_data = []
    for n_train_samples in n_train_samples_list:
        # load the data
        data_flora, data_floral, data_fno_flora, data_fno_floral = load_data(
            n_train_samples
        )
        # get pareto data for fm
        df = ParetoPlot.get_pareto_data(
            data_flora=data_flora, data_floral=data_floral, model="FM"
        )
        all_data.append(df)
        # get pareto data for fno
        df_fno = ParetoPlot.get_pareto_data(
            data_flora=data_fno_flora,
            data_floral=data_fno_floral,
            model="FNO",
            deterministic=True,
        )
        all_data.append(df_fno)
    combined_df = pd.concat(all_data, ignore_index=True)
    display_cols = [
        "Samples (train)",
        "Model",
        "Method",
        "Mean Field Error",
        "Variance Field Error",
    ]
    print(
        combined_df[[c for c in display_cols if c in combined_df.columns]].to_string(
            index=False
        )
    )
    legend_kwargs = {"bbox_to_anchor": (1.01, 0.5), "loc": "center left"}
    ParetoPlot.plot_pareto(
        combined_df,
        ylim_range=(1e-2, 1e1),
        figsize=(8, 5),
        xlim_range=(1e-3, 1e1),
        legend_kwargs=legend_kwargs,
        det_as_lines=True,
    )


def plot_residual_summary(n_train_samples_list):
    all_data = []
    for n_train_samples in n_train_samples_list:
        # load the data
        data_flora, data_floral, data_fno_flora, data_fno_floral = load_data(
            n_train_samples
        )
        # compute the residual for fm
        residual = ResidualOneDCorr(data_flora=data_flora, data_floral=data_floral)
        df = residual(verbose=True, model="FM")
        all_data.append(df)
        # compute the residual for fno
        residual_fno = ResidualOneDCorr(
            data_flora=data_fno_flora, data_floral=data_fno_floral
        )
        df_fno = residual_fno(verbose=True, model="FNO")
        all_data.append(df_fno)
    combined_df = pd.concat(all_data, ignore_index=True)
    ResidualOneDCorr.plot_residual(combined_df, figsize=(7, 5), xlim_range=(1e0, 1e4))


def plot_error_summary(n_train_samples_list):
    all_data = []
    for n_train_samples in n_train_samples_list:
        # load the data
        data_flora, data_floral, data_fno_flora, data_fno_floral = load_data(
            n_train_samples
        )
        # compute error summary for fm
        summary = ErrorSummary(data_flora=data_flora, data_floral=data_floral)
        df = summary(verbose=True, model="FM")
        all_data.append(df)
        # compute error summary for fno
        summary_fno = ErrorSummary(
            data_flora=data_fno_flora, data_floral=data_fno_floral
        )
        df_fno = summary_fno(verbose=True, model="FNO")
        all_data.append(df_fno)
    combined_df = pd.concat(all_data, ignore_index=True)
    ErrorSummary.plot_error(combined_df, ylim_range=(1e-3, 1e0), xlim_range=(0, 1e4))


if __name__ == "__main__":
    # error summary
    plot_error_summary(n_train_samples_list)

    # plot field
    plot_field_idx = None

    if plot_field_idx:
        for ii in plot_field_idx:
            plot_field(n_train_samples=n_train_samples_list[ii])
    else:
        for ii in range(len(n_train_samples_list)):
            plot_field(n_train_samples=n_train_samples_list[ii])

    # pareto
    print_pareto(n_train_samples_list)
