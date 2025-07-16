import torch
import torch.nn as nn
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal

# from gpytorch.constraints import GreaterThan, Interval


def build_gp(train_set, val_set, gp_type: str = "vanilla", **kwargs):
    """function to build a Gaussian Process model based on the specified type."""
    if gp_type == "vanilla":
        return VanillaGP(
            train_set, val_set, device=kwargs.get("device", torch.device("cpu"))
        )
    elif gp_type == "deep_kernel":
        return DeepKernelGP(
            train_set,
            val_set,
            device=kwargs.get("device", torch.device("cpu")),
            hidden_dim=kwargs.get("hidden_dim", 64),
        )
    else:
        raise ValueError(
            f"Unknown GP type: {gp_type}."
            f"Supported types are 'vanilla' and 'deep_kernel'."
        )


class VanillaGP(ExactGP):
    """Vanilla Gaussian Process model for regression tasks."""

    def __init__(self, train_set, val_set, device=None):
        self.train_set = train_set
        self.val_set = val_set

        self.train_x, self.train_y = [t.to(device) for t in train_set.tensors]
        self.val_x, self.val_y = [t.to(device) for t in val_set.tensors]

        # likelihood
        likelihood = GaussianLikelihood()

        # initialize the ExactGP model
        super().__init__(self.train_x, self.train_y.squeeze(), likelihood)

        self.likelihood = likelihood.to(device)

        # mean function
        self.mean_module = ConstantMean()
        # kernel
        self.covar_module = ScaleKernel(RBFKernel())

        # Move to device
        self.to(device)

    def _compute_loss(self, mll, x, y):
        assert y.ndim == 1, "Targets should be 1D."
        output = self.forward(x)
        return -mll(output, y)

    def forward(self, x):
        """Forward pass of the GP model."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def train_step(self, mll, optimizer):
        """Train step for the GP model."""
        self.train()
        self.likelihood.train()
        optimizer.zero_grad()
        loss = self._compute_loss(mll, self.train_x, self.train_y.view(-1))
        if torch.isnan(loss):
            raise RuntimeError("NaN loss during training.")
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def val_step(self, mll):
        self.eval()
        self.likelihood.eval()
        return self._compute_loss(mll, self.val_x, self.val_y.view(-1)).item()


class DeepKernelGP(ExactGP):
    """Deep Kernel Gaussian Process model for regression tasks."""

    def __init__(self, train_set, val_set, device=None, hidden_dim=1):
        self.train_set = train_set
        self.val_set = val_set

        self.train_x, self.train_y = [t.to(device) for t in train_set.tensors]
        self.val_x, self.val_y = [t.to(device) for t in val_set.tensors]

        # likelihood
        likelihood = GaussianLikelihood()

        # initialize the ExactGP model
        super().__init__(self.train_x, self.train_y.squeeze(), likelihood)

        self.likelihood = likelihood.to(device)

        # mean function
        self.mean_module = ConstantMean()
        # feature extractor
        input_dim = self.train_x.shape[1]
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        # kernel
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=hidden_dim // 2))

        # Move to device
        self.to(device)

    def _compute_loss(self, mll, x, y):
        assert y.ndim == 1, "Targets should be 1D."
        output = self.forward(x)
        return -mll(output, y)

    def forward(self, x):
        """Forward pass of the GP model."""
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)

    def train_step(self, mll, optimizer):
        """Train step for the GP model."""
        self.train()
        self.likelihood.train()
        optimizer.zero_grad()
        loss = self._compute_loss(mll, self.train_x, self.train_y.view(-1))
        if torch.isnan(loss):
            raise RuntimeError("NaN loss during training.")
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def val_step(self, mll):
        self.eval()
        self.likelihood.eval()
        return self._compute_loss(mll, self.val_x, self.val_y.view(-1)).item()
