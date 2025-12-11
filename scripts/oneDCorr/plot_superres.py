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
n_train_samples = 10
n_val_samples = 1000
train_res_list = [8, 16, 32, 64]
results_folder = "./results_data"
# End user input

plt.style.use("../journal.mplstyle")


def combine_path(paths: list):
    return osp.join(*paths)


def load_data(train_res):
    nt = n_train_samples
    nv = n_val_samples
    assert isinstance(train_res, (str, int))
    res = train_res.strip().lower() if isinstance(train_res, str) else train_res
    file_flora = f"onedcorr_n_train_{nt}_n_val_{nv}_train_res_{res}_results_flora.pt"
    file_floral = f"onedcorr_n_train_{nt}_n_val_{nv}_train_res_{res}_results_floral.pt"
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


def plot_field(train_res):
    save_name = f"field_samples_n_train_{n_train_samples}"
    f"_n_val_{n_val_samples}_train_res_{train_res}"
    # load the data
    data_flora, data_floral = load_data(train_res)
    # create plot object
    plotter = oneDPlot(data_flora=data_flora, data_floral=data_floral)
    # create sample plot
    plotter.make_field_sample_plot(std_factor=10, save_name=save_name)


def plot_pareto(train_res_list):
    all_data = []
    for train_res in train_res_list:
        # load the data
        data_flora, data_floral = load_data(train_res)
        # get pareto data
        df = ParetoPlot.get_pareto_data(data_flora=data_flora, data_floral=data_floral)
        # add the train res
        df["Resolution (train)"] = train_res
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    ParetoPlot.plot_pareto_res(combined_df, ylim_range=(1e-2, 1e1), figsize=(7, 5))


def plot_error_summary(train_res_list):
    all_data = []
    for train_res in train_res_list:
        # load the data
        data_flora, data_floral = load_data(train_res)
        # get pareto data
        summary = ErrorSummary(data_flora=data_flora, data_floral=data_floral)
        df = summary(verbose=True)
        # add the train res
        df["Resolution (train)"] = train_res
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    ErrorSummary.plot_error_vs_train_res(
        combined_df, ylim_range=(1e-2, 1e0), xlim_range=(1e0, 1e2)
    )


if __name__ == "__main__":
    # error summary
    # plot_error_summary(train_res_list)
    # plot field
    # plot_field(train_res=train_res_list[-1])
    # plot_field(train_res=train_res_list[0])
    # # pareto
    plot_pareto(train_res_list)
