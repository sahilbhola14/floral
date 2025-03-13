import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP class"""

    def __init__(
        self,
        in_dim=1,
        width=[32, 32, 32],
        out_dim=1,
        activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
    ):
        super(MLP, self).__init__()
        """ init MLP """
        assert (
            len(width) == len(activations) - 1
        ), "Output activation must also be specified (None for no activation)"
        layers = []
        for ii, jj in zip([in_dim] + width[:-1], width):
            layers.extend([nn.Linear(ii, jj), activations.pop(0)])
        layers.append(nn.Linear(width[-1], out_dim))
        if activations[0] is not None:
            layers.append(activations[0])
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        """forward pass"""
        return self.net(input)


class FiLM(nn.Module):
    """FiLM"""

    def __init__(self, in_dim: int, out_dim: int):
        super(FiLM, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gamma = MLP(
            in_dim, [64, 64], out_dim, activations=[nn.ReLU(), nn.ReLU(), None]
        )
        self.beta = MLP(in_dim, [64, 64], 1, activations=[nn.ReLU(), nn.ReLU(), None])

    def forward(self, x: torch.Tensor, mod: torch.Tensor):
        """forward pass"""
        gamma = self.gamma(mod)
        beta = self.beta(mod)
        return x @ gamma.T + beta.T
