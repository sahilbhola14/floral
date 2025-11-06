"""
Plotting utilities
"""

import torch
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC


def find_best_idx(true, pred):
    """Find the best sample index based on L1 error"""
    assert true.shape == pred.shape
    # compute error for each sample (n_samples,)
    error = (true - pred).abs().flatten(1).mean(1)
    idx_best = error.argmin().item()
    print(f"Best sample idx: {idx_best}, error: {error[idx_best]:.4e}")
    return idx_best


class BasePlot(ABC):

    def make_pareto_plot(self):
        pass

    @abstractmethod
    def make_field_sample_plot(self):
        """plot mean fields"""
        pass

    @abstractmethod
    def make_field_plot(self):
        pass

    @abstractmethod
    def make_error_sample_plot(self):
        pass

    @abstractmethod
    def make_error_plot(self):
        pass


class twoDPlot(BasePlot):
    """Plot multiple samples"""

    def __init__(self, data_flora: dict, data_floral: dict):
        super(twoDPlot, self).__init__()
        self.data_flora = data_flora
        self.data_floral = data_floral

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

        # compute mean and std over the UQ samples (B, C, *dims)
        self.mean_dict = {
            "Low-fidelity": self.LF_field,
            "High-fidelity": self.HF_field,
            "FLORA": self.HF_field_prediction_flora.mean(1),
            "FLORAL": self.HF_field_prediction_floral.mean(1),
        }

        self.std_dict = {
            "Low-fidelity": torch.ones_like(self.LF_field) * 1e-6,
            "FLORA": self.HF_field_prediction_flora.std(1),
            "FLORAL": self.HF_field_prediction_floral.std(1),
        }

    def make_field_sample_plot(
        self, n_samples: int = 5, plot_channel: int = 0, **kwargs
    ):
        """make field samples plot"""
        assert n_samples <= self.n_avail_samples
        print(f"Plotting {n_samples} samples for channel: {plot_channel}")
        fig, axs = plt.subplots(
            n_samples,
            len(self.mean_dict),
            figsize=kwargs.get("figsize", (4, 6)),
            dpi=300,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        # compute range
        vmin = min(data[:n_samples].min() for data in self.mean_dict.values())
        vmax = max(data[:n_samples].max() for data in self.mean_dict.values())
        vmin = torch.floor(vmin)
        vmax = torch.ceil(vmax)
        for jj, k in enumerate(self.mean_dict.keys()):
            axs[0, jj].set_title(k)
            for ii in range(n_samples):
                im = axs[ii, jj].imshow(
                    self.mean_dict[k][ii][plot_channel],
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                    interpolation="bicubic",
                )
        fig.colorbar(im, ax=axs[-1], orientation="horizontal", pad=0.1)
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(kwargs.get("xlabel", r"$x_1$"))
            ax.set_ylabel(kwargs.get("ylabel", r"$x_2$"))
            ax.label_outer()
        plt.savefig("test.png")

    def make_field_plot(self):
        pass

    def make_error_sample_plot(self):
        pass

    def make_error_plot(self):
        pass
