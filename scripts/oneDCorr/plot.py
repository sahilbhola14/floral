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
n_train_samples_list = [100, 200, 2000]
n_val_samples = 750
n_test_samples = 750
operator_method = "filmfno"
train_res = "Full"
results_folder = "./results_vary_training_size_grf_input_multi_sample_a"
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
    plotter.make_field_sample_plot(figsize=(15, 12), linewidth=3.5, label_fontsize=20)


def plot_pareto(n_train_samples_list):
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
    legend_kwargs = {"bbox_to_anchor": (1.01, 0.5), "loc": "center left"}
    ParetoPlot.plot_pareto(
        combined_df,
        ylim_range=(1e-2, 1e1),
        figsize=(8, 5),
        xlim_range=(1e-2, 1e0),
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
    ErrorSummary.plot_error(combined_df, ylim_range=(1e-3, 1e1), xlim_range=(0, 1e4))


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
    plot_pareto(n_train_samples_list)
