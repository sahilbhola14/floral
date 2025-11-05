"""
Plotting utilities
"""

import torch
import matplotlib.pyplot as plt

# import seaborn as sns

plt.style.use("../../../scripts/journal.mplstyle")


def find_best_idx(true, pred):
    """Find the best sample index based on L1 error"""
    assert true.shape == pred.shape
    # compute error for each sample (n_samples,)
    error = (true - pred).abs().flatten(1).mean(1)
    idx_best = error.argmin().item()
    print(f"Best sample idx: {idx_best}, error: {error[idx_best]:.4e}")
    return idx_best


class twoDPlot:
    """Plot multiple samples"""

    def __init__(self, data_flora: dict, data_floral: dict, n_samples: int = 10):
        self.data_flora = data_flora
        self.data_floral = data_floral
        self.n_samples = n_samples  # number of samples to plot

        # High-fidelity data (B, channels, *dims)
        self.HF_field = self.data_floral["HF_field_plot"]
        # Low-fidelity data (B, channels, *dims)
        self.LF_field = self.data_floral["LF_field_plot"]
        # Prediction Flora
        self.HF_field_prediction_flora = self.data_flora["HF_field_prediction_plot"]
        # Prediction Floral
        self.HF_field_prediction_floral = self.data_floral["HF_field_prediction_plot"]

        self.n_avail_samples = len(self.HF_field)
        assert len(self.LF_field) == self.n_avail_samples
        assert len(self.HF_field_prediction_flora) == self.n_avail_samples
        assert len(self.HF_field_prediction_floral) == self.n_avail_samples

        # number of UQ samples
        self.n_gen = self.HF_field_prediction_floral.shape[1]
        assert self.n_gen == self.HF_field_prediction_flora.shape[1]

        # compute mean and std
        self.mean_dict = {
            "High-fidelity": self.HF_field,
            "Low-fideltiy": self.LF_field,
            "FLORA": self.HF_field_prediction_flora.mean(1),
            "FLORAL": self.HF_field_prediction_floral.mean(1),
        }

        self.std_dict = {
            "Low-fideltiy": torch.ones_like(self.LF_field) * 1e-6,
            "FLORA": self.HF_field_prediction_flora.std(1),
            "FLORAL": self.HF_field_prediction_floral.std(1),
        }

    def make_field_sample_plot(self):
        """make multi-sample plot"""
        pass
