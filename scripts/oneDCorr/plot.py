# scripts/onedcorr/plot.py
"""
OneDCorr flow plots
"""
import pandas as pd
import torch
import matplotlib.pyplot as plt
from floral.utils import oneDPlot, ParetoPlot, ErrorSummary, BaseResidual

# Begin user input
n_train_samples_list = [50, 100, 1000, 5000]
n_val_samples = 1000
# End user input

plt.style.use("../journal.mplstyle")


def load_data(n_train_samples):
    nt = n_train_samples
    nv = n_val_samples
    file_flora = f"onedcorr_n_train_{nt}_n_val_{nv}_results_flora.pt"
    file_floral = f"onedcorr_n_train_{nt}_n_val_{nv}_results_floral.pt"
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
    data_flora, data_floral = load_data(n_train_samples)
    # create plot object
    plotter = oneDPlot(data_flora=data_flora, data_floral=data_floral)
    # create sample plot
    plotter.make_field_sample_plot(std_factor=20)


def plot_pareto(n_train_samples_list):
    all_data = []
    for n_train_samples in n_train_samples_list:
        # load the data
        data_flora, data_floral = load_data(n_train_samples)
        # get pareto data
        df = ParetoPlot.get_pareto_data(data_flora=data_flora, data_floral=data_floral)
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    ParetoPlot.plot_pareto(combined_df, ylim_range=(1e-2, 1e1))


def plot_residual_summary(n_train_samples_list):
    all_data = []
    for n_train_samples in n_train_samples_list:
        # load the data
        data_flora, data_floral = load_data(n_train_samples)
        # compute the residual
        residual = ResidualOneDCorr(data_flora=data_flora, data_floral=data_floral)
        df = residual(verbose=True)
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    ResidualOneDCorr.plot_residual(combined_df)


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
    ErrorSummary.plot_error(combined_df, ylim_range=(1e-4, 1e1))


if __name__ == "__main__":
    # residual summary
    plot_residual_summary(n_train_samples_list)
    # error summary
    plot_error_summary(n_train_samples_list)
    # # plot field
    plot_field(n_train_samples=n_train_samples_list[0])
    # # pareto
    plot_pareto(n_train_samples_list)
