import torch
import torch.nn as nn
from mfFlow.archs import MLP


class RBFFiLM(nn.Module):
    """Radial Basis Function FiLM (Good for encodding spatial information)
    TODO:
    - Make this amenable to multiple dimensions of domain.
    """

    def __init__(self, latent_dim: int, num_centers: int, nd: int):
        """
        Args:
            latent_dim (int): Dimension of the latent space
            num_centers (int): Number of radial basis function centers
            nd (int): Number of dimensions of the `mod` tensor
        """
        super(RBFFiLM, self).__init__()
        self.num_centers = num_centers
        self.latent_dim = latent_dim
        self.nd = nd
        assert self.nd <= 2, "RBF FiLM only supports 1D or 2D modulations."

        # Lernable centers for the RBF encoder
        self.centers = nn.Parameter(
            torch.stack(
                [torch.linspace(0, 1, self.num_centers) for _ in range(self.nd)], dim=-1
            ),
            requires_grad=True,
        )

        self.gamma_net = MLP(
            in_dim=self.num_centers,
            width=[64, 64],
            out_dim=self.latent_dim,
            activations=[nn.ReLU(), nn.ReLU(), None],
        )

        self.beta_net = MLP(
            in_dim=self.num_centers,
            width=[64, 64],
            out_dim=1,
            activations=[nn.ReLU(), nn.ReLU(), None],
        )

    def _rbf_encoding(self, mod, gamma=10.0):
        """Radial Basis Function encoding

        Computes the RBF encoding of the modulation tensor
        Args:
            mod (torch.Tensor): Modulation tensor of shape (batch_size, nd)
            gamma (float): Hyperparameter for the RBF kernel
        Returns:
            torch.Tensor: RBF encoded tensor of shape (batch_size, num_centers)
        """
        # compute the Delta (batch_size, num_centers, nd)
        delta = mod.unsqueeze(1) - self.centers.unsqueeze(0)
        # compute the RBF kernel
        return torch.exp(-gamma * torch.norm(delta, dim=-1) ** 2)

    def forward(self, x: torch.Tensor, mod: torch.Tensor):
        """forward pass of the FiLM layer
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim)
            mod (torch.Tensor): Modulation tensor of shape (batch_size, in_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, nx)
        """
        # encode the mod features
        mod_features = self._rbf_encoding(mod)
        gamma = self.gamma_net(mod_features)
        beta = self.beta_net(mod_features)
        return torch.sum(x * gamma, dim=-1, keepdim=True) + beta


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
