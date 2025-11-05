"""
Plotting utilities
"""


class SamplePlot:
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
