""" One-D correlation solver
Reference: Thakur, A., Tripura, T. and Chakraborty, S., 2022. Multi-fidelity wavelet
neural operator with application to uncertainty quantification.
arXiv preprint arXiv:2208.05606.
"""

import torch
import torch.nn as nn


def eval_high_fidelity_model(a: torch.Tensor, x: torch.Tensor):
    """High fidelity model: sin(a)"""
    return torch.sin(a)


def eval_low_fidelity_model(a: torch.Tensor, x: torch.Tensor):
    """Low fidelity model: sin(a) + x - 0.25*a"""
    return torch.sin(a) + x.reshape(1, -1) - 0.25 * a


class OneDCorr(nn.Module):
    def __init__(
        self,
        fidelity: str = "High",
        resolution: int = 128,
        n_samples: int = 10,
        k_range: list = [10, 14],
    ):
        super(OneDCorr, self).__init__()
        assert fidelity in ["High", "Low"]
        self.fidelity = fidelity
        self.solver = self._get_solver()
        self.resolution = resolution
        self.n_samples = n_samples
        self.k_range = k_range

        self.domain = torch.linspace(0, 1, self.resolution).view(1, -1)

    def _get_solver(self):
        if self.fidelity == "High":
            return eval_high_fidelity_model
        elif self.fidelity == "Low":
            return eval_low_fidelity_model

    def _sample_input_function(self):
        delta_k = self.k_range[1] - self.k_range[0]
        k = (torch.rand(self.n_samples, 1) * delta_k) + self.k_range[0]
        a = k * self.domain - 4.0
        assert a.shape == (self.n_samples, self.resolution)
        return a

    def _check_input(self, a):
        assert isinstance(a, torch.Tensor) and a.ndim == 2, "a must be a 2D tensor, "
        " got shape {a.shape if isinstance(a, torch.Tensor) else 'not a tensor'}"
        assert (
            a.shape[-1] == self.resolution
        ), f"a must have shape (N, {self.resolution}), got {a.shape}"

    def forward(self, a: torch.Tensor):
        self._check_input(a)
        result = self.solver(a=a, x=self.domain)
        assert result.shape == (self.n_samples, self.resolution)
        return result
