""" One-D correlation solver
Modified from Thakur, A., Tripura, T. and Chakraborty, S., 2022. Multi-fidelity wavelet
neural operator with application to uncertainty quantification.
arXiv preprint arXiv:2208.05606.
"""

import torch
import torch.nn as nn


class OneDCorr(nn.Module):
    def __init__(
        self,
        resolution: int = 128,
        n_samples: int = 10,
        k_range: list = [10, 14],
        seed: int = None,
    ):
        super(OneDCorr, self).__init__()
        self.resolution = resolution
        self.n_samples = n_samples
        self.k_range = k_range
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        self.domain = torch.linspace(0, 1, self.resolution).view(1, -1)

    def _sample_input_function(self):
        delta_k = self.k_range[1] - self.k_range[0]
        k = (
            torch.rand(self.n_samples, 1, generator=self.rng) * delta_k
        ) + self.k_range[0]
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
        result = torch.sin(a) + self.domain.reshape(1, -1) - 0.25 * a
        assert result.shape == (len(a), self.resolution)
        return result
