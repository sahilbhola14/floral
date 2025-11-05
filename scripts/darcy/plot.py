# scripts/darcy/plot.py
"""
Plot HF mean removed 2D fields.
"""
import torch

# from floral.utils import twoDPlot

# Begin user input
n_train_samples_list = [500]
n_val_samples = 500
plot_idx = {"automatic": False, "idx": 32}  # Set to True for automatic index selection
plot_error = False  # Whether to plot error or not
# End user input


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
    # plotter = twoDPlot(data_flora, data_floral)
    raise NotImplementedError("Plotting function is not yet implemented.")


if __name__ == "__main__":
    # plot field
    plot_field()
