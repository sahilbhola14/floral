# scripts/darcy/plot.py
"""
Darcy flow plots
"""
import pandas as pd
import torch
import matplotlib.pyplot as plt
from floral.utils import twoDPlot, ParetoPlot

# Begin user input
n_train_samples_list = [10000]
n_val_samples = 1000
# End user input

plt.style.use("../journal.mplstyle")


def load_data(n_train_samples):
    nt = n_train_samples
    nv = n_val_samples
    file_flora = f"darcy_n_train_{nt}_n_val_{nv}_results_flora.pt"
    file_floral = f"darcy_n_train_{nt}_n_val_{nv}_results_floral.pt"
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


def plot_field(n_train_samples):
    # load the data
    data_flora, data_floral = load_data(n_train_samples)
    # create plot object
    plotter = twoDPlot(data_flora=data_flora, data_floral=data_floral)
    # create sample plot
    plotter.make_field_sample_plot()
    # create sample error plot
    plotter.make_error_sample_plot()


def plot_pareto(n_train_samples_list):
    all_data = []
    for n_train_samples in n_train_samples_list:
        # load the data
        data_flora, data_floral = load_data(n_train_samples)
        # get pareto data
        df = ParetoPlot.get_pareto_data(data_flora=data_flora, data_floral=data_floral)
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    ParetoPlot.plot_pareto(combined_df)


if __name__ == "__main__":
    # plot field
    plot_field(n_train_samples=n_train_samples_list[0])
    # pareto
    plot_pareto(n_train_samples_list)
