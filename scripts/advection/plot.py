# scripts/advection/plot.py
"""
Advection flow plots
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

ADVECTION_SPEED = 0.05
SPATIAL_DOMAIN = (0, 1)
TEMPORAL_DOMAIN = (0, 1)


def combine_path(paths: list):
    return osp.join(*paths)


def load_data(n_train_samples):
    nt = n_train_samples
    nv = n_val_samples
    file_flora = f"advection_n_train_{nt}_n_val_{nv}_results_flora.pt"
    file_floral = f"advection_n_train_{nt}_n_val_{nv}_results_floral.pt"
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


class ResidualAdvection(BaseResidual):
    def __init__(self, data_flora, data_floral):
        super(ResidualAdvection, self).__init__(
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

        # dudt (forward difference + backward difference)
        dudt = torch.zeros_like(prediction)
        dudt[:, :-1, :] = (prediction[:, 1:, :] - prediction[:, :-1, :]) / dt
        dudt[:, -1, :] = (prediction[:, -1, :] - prediction[:, -2, :]) / dt
        # dudx (upwinding)
        if ADVECTION_SPEED > 0:
            dudx = (prediction - torch.roll(prediction, shifts=1, dims=2)) / dx
        else:
            dudx = (torch.roll(prediction, shifts=-1, dims=2)) / dx
        # residual (Nt, Nx)
        residual = dudt + ADVECTION_SPEED * dudx

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
        residual = ResidualAdvection(data_flora=data_flora, data_floral=data_floral)
        df = residual(verbose=True)
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    ResidualAdvection.plot_residual(combined_df, figsize=(7, 5), ylim_range=(1e-3, 1e2))


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
