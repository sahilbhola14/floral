# scripts/darcy/plot_superres.py
"""
Darcy super-resolution flow plots (vary training resolution, fixed n_train)
"""
import os.path as osp
import pandas as pd
import torch
import matplotlib.pyplot as plt
from floral.utils import twoDPlot, ParetoPlot, ErrorSummary, BaseResidual

# Begin user input
n_train_samples = 500
n_val_samples = 750
n_test_samples = 750
operator_method = "filmfno"
train_res_list = [8, 16]
results_folder = "./results_superres"
# End user input

plt.style.use("../journal.mplstyle")


def combine_path(paths: list):
    return osp.join(*paths)


def load_data(train_res):
    nt = n_train_samples
    nv = n_val_samples
    ntest = n_test_samples
    method = operator_method.lower().strip()
    assert isinstance(train_res, (str, int))
    res = train_res.strip().lower() if isinstance(train_res, str) else train_res

    file_flora = (
        f"darcy_fm_operator_method_{method}_n_train_{nt}_n_val_{nv}"
        f"_n_test_{ntest}_train_res_{res}_results_flora.pt"
    )
    file_floral = (
        f"darcy_fm_operator_method_{method}_n_train_{nt}_n_val_{nv}"
        f"_n_test_{ntest}_train_res_{res}_results_floral.pt"
    )
    file_fno_flora = (
        f"darcy_fno_operator_method_{method}_n_train_{nt}_n_val_{nv}"
        f"_n_test_{ntest}_train_res_{res}_results_flora.pt"
    )
    file_fno_floral = (
        f"darcy_fno_operator_method_{method}_n_train_{nt}_n_val_{nv}"
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
        "Number of samples for UQ (FNO Flora): "
        f"{data_fno_flora['prediction'].shape[1]}"
    )
    print(
        "Number of samples for UQ (FNO Floral): "
        f"{data_fno_floral['prediction'].shape[1]}"
    )
    print("--" * 10)

    return data_flora, data_floral, data_fno_flora, data_fno_floral


class ResidualDarcy(BaseResidual):
    def __init__(self, data_flora, data_floral):
        super(ResidualDarcy, self).__init__(
            data_flora=data_flora, data_floral=data_floral
        )
        raise NotImplementedError("Darcy residual not currently implemented")


def plot_field(train_res):
    save_name = (
        f"field_samples_n_train_{n_train_samples}"
        f"_n_val_{n_val_samples}_train_res_{train_res}"
    )
    data_flora, data_floral, data_fno_flora, data_fno_floral = load_data(train_res)
    plotter = twoDPlot(
        data_flora=data_flora,
        data_floral=data_floral,
        data_fno_flora=data_fno_flora,
        data_fno_floral=data_fno_floral,
    )
    plotter.make_field_sample_plot(
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        n_samples=4,
        save_name=save_name,
    )
    plotter.make_mean_std_sample_plot(
        state_name=r"u",
        sample_idx=0,
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        figsize=(len(plotter.mean_dict) * 2.5, 5.0),
        save_name=(
            f"mean_std_n_train_{n_train_samples}_n_val_{n_val_samples}"
            f"_train_res_{train_res}"
        ),
    )


def print_pareto(train_res_list):
    all_data = []
    for train_res in train_res_list:
        data_flora, data_floral, data_fno_flora, data_fno_floral = load_data(train_res)
        df = ParetoPlot.get_pareto_data(
            data_flora=data_flora, data_floral=data_floral, model="FM"
        )
        df["Resolution (train)"] = train_res
        all_data.append(df)
        df_fno = ParetoPlot.get_pareto_data(
            data_flora=data_fno_flora, data_floral=data_fno_floral, model="FNO"
        )
        df_fno["Resolution (train)"] = train_res
        all_data.append(df_fno)
    combined_df = pd.concat(all_data, ignore_index=True)
    display_cols = [
        "Samples (train)",
        "Resolution (train)",
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


def plot_error_summary(train_res_list):
    all_data = []
    for train_res in train_res_list:
        data_flora, data_floral, data_fno_flora, data_fno_floral = load_data(train_res)
        summary = ErrorSummary(data_flora=data_flora, data_floral=data_floral)
        df = summary(verbose=True, model="FM")
        df["Resolution (train)"] = train_res
        all_data.append(df)
        summary_fno = ErrorSummary(
            data_flora=data_fno_flora, data_floral=data_fno_floral
        )
        df_fno = summary_fno(verbose=True, model="FNO")
        df_fno["Resolution (train)"] = train_res
        all_data.append(df_fno)
    combined_df = pd.concat(all_data, ignore_index=True)
    ErrorSummary.plot_error_vs_train_res(
        combined_df, ylim_range=(1e-2, 1e1), xlim_range=(1e0, 1e2)
    )


def plot_residual_summary(train_res_list):
    raise NotImplementedError("Darcy residual not currently implemented")


if __name__ == "__main__":
    # error summary
    plot_error_summary(train_res_list)
    # plot field
    for ii in range(len(train_res_list)):
        plot_field(train_res=train_res_list[ii])
    # pareto
    print_pareto(train_res_list)
