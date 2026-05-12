# scripts/burgers/plot_superres.py
"""
Burgers super-resolution flow plots (vary training resolution, fixed n_train)
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

VISCOSITY = 0.01
SPATIAL_DOMAIN = (0, 1)
TEMPORAL_DOMAIN = (0, 0.2)


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
        f"burgers_fm_operator_method_{method}_n_train_{nt}_n_val_{nv}"
        f"_n_test_{ntest}_train_res_{res}_results_flora.pt"
    )
    file_floral = (
        f"burgers_fm_operator_method_{method}_n_train_{nt}_n_val_{nv}"
        f"_n_test_{ntest}_train_res_{res}_results_floral.pt"
    )
    file_fno_flora = (
        f"burgers_fno_operator_method_{method}_n_train_{nt}_n_val_{nv}"
        f"_n_test_{ntest}_train_res_{res}_results_flora.pt"
    )
    file_fno_floral = (
        f"burgers_fno_operator_method_{method}_n_train_{nt}_n_val_{nv}"
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


class ResidualBurgers(BaseResidual):
    def __init__(self, data_flora, data_floral):
        super(ResidualBurgers, self).__init__(
            data_flora=data_flora, data_floral=data_floral
        )
        self.condition = self.full_condition[
            :, 0
        ]  # for burgers, only the first channel is the condition

    def comp_residual(self, prediction, condition, domain):  # type: ignore[override]
        """compute the residual for the Burgers equation
        prediction: (batch_size, Nt, Nx)
        """
        x = domain[0][0]  # (Nx,)
        t = domain[1][:, 0]  # (Nt,)

        dx = x[1] - x[0]
        dt = t[1] - t[0]

        # dudx (upwind, second-order)
        pos_mask = (prediction > 0).float()
        neg_mask = 1.0 - pos_mask
        u_ip1 = torch.roll(prediction, shifts=-1, dims=2)
        u_ip2 = torch.roll(prediction, shifts=-2, dims=2)
        u_im1 = torch.roll(prediction, shifts=1, dims=2)
        u_im2 = torch.roll(prediction, shifts=2, dims=2)

        dudx_pos = (3 * prediction - 4 * u_im1 + u_im2) / (2 * dx)
        dudx_neg = (-3 * prediction + 4 * u_ip1 - u_ip2) / (2 * dx)

        dudx = pos_mask * dudx_pos + neg_mask * dudx_neg

        # d2udx2
        d2udx2 = (u_ip1 - 2.0 * prediction + u_im1) / (dx * dx)

        # dudt
        dudt = torch.zeros_like(prediction)
        dudt[:, :-1, :] = (prediction[:, 1:, :] - prediction[:, :-1, :]) / dt
        dudt[:, -1, :] = (prediction[:, -1, :] - prediction[:, -2, :]) / dt

        return dudt + prediction * dudx - VISCOSITY * d2udx2


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
        xlabel=r"$x$",
        ylabel=r"$t$",
        n_samples=4,
        save_name=save_name,
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
    all_data = []
    for train_res in train_res_list:
        data_flora, data_floral, data_fno_flora, data_fno_floral = load_data(train_res)
        residual = ResidualBurgers(data_flora=data_flora, data_floral=data_floral)
        df = residual(verbose=True, model="FM")
        df["Resolution (train)"] = train_res
        all_data.append(df)
        residual_fno = ResidualBurgers(
            data_flora=data_fno_flora, data_floral=data_fno_floral
        )
        df_fno = residual_fno(verbose=True, model="FNO")
        df_fno["Resolution (train)"] = train_res
        all_data.append(df_fno)
    combined_df = pd.concat(all_data, ignore_index=True)
    ResidualBurgers.plot_residual(combined_df, figsize=(7, 5), ylim_range=(1e-3, 1e2))


if __name__ == "__main__":
    # error summary
    # plot_error_summary(train_res_list)
    # plot field
    # for ii in range(len(train_res_list)):
    #     plot_field(train_res=train_res_list[ii])
    # pareto
    print_pareto(train_res_list)
