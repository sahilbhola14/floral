# scripts/burgers/plot.py
"""
Burgers flow plots
"""
import os.path as osp
import pandas as pd
import torch
import matplotlib.pyplot as plt
from floral.utils import twoDPlot, ParetoPlot, ErrorSummary, BaseResidual

# Begin user input
n_train_samples_list = [500, 1000, 5000]
n_val_samples = 1000
results_folder = "./results_data"
# End user input

plt.style.use("../journal.mplstyle")

VISCOSITY = 0.01
SPATIAL_DOMAIN = (0, 1)
TEMPORAL_DOMAIN = (0, 0.2)


def combine_path(paths: list):
    return osp.join(*paths)


def load_data(n_train_samples):
    nt = n_train_samples
    nv = n_val_samples
    file_flora = f"burgers_n_train_{nt}_n_val_{nv}_results_flora.pt"
    file_floral = f"burgers_n_train_{nt}_n_val_{nv}_results_floral.pt"
    print("flora file: ", file_flora)
    print("floral file: ", file_floral)
    print("n_train_samples: ", n_train_samples)
    print("n_val_samples: ", n_val_samples)
    data_flora = torch.load(
        combine_path([results_folder, file_flora]), weights_only=False
    )
    data_floral = torch.load(
        combine_path([results_folder, file_floral]), weights_only=False
    )
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


class ResidualBurgers(BaseResidual):
    def __init__(self, data_flora, data_floral):
        super(ResidualBurgers, self).__init__(
            data_flora=data_flora, data_floral=data_floral
        )
        # condition
        self.condition = self.full_condition[
            :, 0
        ]  # for onedcorr, only the first channel is the condition

    def comp_residual(self, prediction, condition, domain):
        """compute the residual for the advection equation
        prediciton: (batch_size, Nt, Nx)
        """
        # Unpack space / time grids
        x = domain[0][0]  # (Nx,)
        t = domain[1][:, 0]  # (Nt,)

        dx = x[1] - x[0]
        dt = t[1] - t[0]

        # dudx
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

        # dtdt
        dudt = torch.zeros_like(prediction)
        dudt[:, :-1, :] = (prediction[:, 1:, :] - prediction[:, :-1, :]) / dt
        dudt[:, -1, :] = (prediction[:, -1, :] - prediction[:, -2, :]) / dt

        # residual (Nt, Nx)
        residual = dudt + prediction * dudx - VISCOSITY * d2udx2

        return residual


def plot_field(n_train_samples):
    # load the data
    data_flora, data_floral = load_data(n_train_samples)
    # create plot object
    plotter = twoDPlot(data_flora=data_flora, data_floral=data_floral)
    # create sample plot
    plotter.make_field_sample_plot(xlabel=r"$x$", ylabel=r"$t$", n_samples=4)
    # create sample error plot
    plotter.make_error_sample_plot(
        xlabel=r"$x$", ylabel=r"$t$", n_samples=4, vmin=0, vmax=0.25
    )


def plot_pareto(n_train_samples_list):
    all_data = []
    for n_train_samples in n_train_samples_list:
        # load the data
        data_flora, data_floral = load_data(n_train_samples)
        # get pareto data
        df = ParetoPlot.get_pareto_data(data_flora=data_flora, data_floral=data_floral)
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    ParetoPlot.plot_pareto(combined_df, figsize=(7, 5))


def plot_error_summary(n_train_samples_list):
    all_data = []
    for n_train_samples in n_train_samples_list:
        # load the data
        data_flora, data_floral = load_data(n_train_samples)
        # get pareto data
        summary = ErrorSummary(data_flora=data_flora, data_floral=data_floral)
        df = summary(verbose=True)
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    ErrorSummary.plot_error(combined_df, ylim_range=(1e-4, 1e2), xlim_range=(1e2, 1e4))


def plot_residual_summary(n_train_samples_list):
    all_data = []
    for n_train_samples in n_train_samples_list:
        # load the data
        data_flora, data_floral = load_data(n_train_samples)
        # compute the residual
        residual = ResidualBurgers(data_flora=data_flora, data_floral=data_floral)
        df = residual(verbose=True)
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    ResidualBurgers.plot_residual(combined_df, figsize=(7, 5), ylim_range=(1e-3, 1e2))


if __name__ == "__main__":
    # residual summary
    plot_residual_summary(n_train_samples_list)
    # error summary
    plot_error_summary(n_train_samples_list)
    # plot field
    plot_field(n_train_samples=n_train_samples_list[0])
    plot_field(n_train_samples=n_train_samples_list[-1])
    # pareto
    plot_pareto(n_train_samples_list)
