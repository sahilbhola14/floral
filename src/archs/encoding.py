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


class RBFFiLM(nn.Module):
    """Radial Basis Function FiLM (Good for encodding spatial information)"""

    def __init__(self, in_dim: int, out_dim: int, num_centers: int = 10):
        super(RBFFiLM, self).__init__()
        self.num_centers = num_centers
        self.centers = nn.Parameter(
            torch.linspace(0, 1, self.num_centers)
        )  # Learnable centers
        self.gamma_net = MLP(
            self.num_centers,
            [64, 64],
            out_dim,
            activations=[nn.ReLU(), nn.ReLU(), None],
        )
        self.beta_net = MLP(
            self.num_centers, [64, 64], 1, activations=[nn.ReLU(), nn.ReLU(), None]
        )

    def rbf_encoding(self, mod, gamma=10.0):
        return torch.exp(-gamma * (mod - self.centers) ** 2)

    def forward(self, x: torch.Tensor, mod: torch.Tensor):
        mod_features = self.rbf_encoding(mod)
        gamma = self.gamma_net(mod_features)
        beta = self.beta_net(mod_features)
        return torch.einsum("bd,od->bo", x, gamma) + beta.T


class FiLM(nn.Module):
    """FiLM"""

    def __init__(self, in_dim: int, out_dim: int, n_freq: int):
        super(FiLM, self).__init__()
        self.in_dim = in_dim
        self.n_freq = n_freq
        self.out_dim = out_dim
        self.gamma = MLP(
            self.in_dim * self.n_freq * 2,
            [64, 64],
            self.out_dim,
            activations=[nn.ReLU(), nn.ReLU(), None],
        )
        self.beta = MLP(
            self.in_dim * self.n_freq * 2,
            [64, 64],
            1,
            activations=[nn.ReLU(), nn.ReLU(), None],
        )

    def forward(self, x: torch.Tensor, mod: torch.Tensor):
        """forward pass"""
        f = 2 * torch.arange(1, self.n_freq + 1, device=x.device) * torch.pi
        mod = torch.cat([(f * mod).sin(), (f * mod).cos()], dim=-1)
        gamma = self.gamma(mod)
        beta = self.beta(mod)
        return x @ gamma.T + beta.T
