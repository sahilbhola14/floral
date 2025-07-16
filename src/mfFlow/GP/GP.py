import torch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import GreaterThan, Interval


class GPRegressionModel(ExactGP):
    """Gaussian Process model for regression tasks."""

    def __init__(self, train_set, val_set, device=None):
        self.train_set = train_set
        self.val_set = val_set

        self.train_x, self.train_y = [t.to(device) for t in train_set.tensors]
        self.val_x, self.val_y = [t.to(device) for t in val_set.tensors]

        # Likelihood with constrained noise
        likelihood = GaussianLikelihood()
        likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-4))

        super().__init__(self.train_x, self.train_y.view(-1), likelihood)
        self.likelihood = likelihood.to(device)

        # Mean and kernel with constraints
        self.mean_module = ConstantMean()
        base_kernel = RBFKernel()
        base_kernel.register_constraint("raw_lengthscale", Interval(0.05, 5.0))
        self.covar_module = ScaleKernel(base_kernel)

        # Move to device
        self.to(device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def _compute_loss(self, mll, x, y):
        assert y.ndim == 1, "Targets should be 1D."
        output = self.forward(x)
        return -mll(output, y)

    def train_step(self, mll, optimizer):
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

    def print_params(self):
        print(
            "[params]"
            f"Lengthscale: {self.covar_module.base_kernel.lengthscale.item(): .4f}, "
            f"Outputscale: {self.covar_module.outputscale.item(): .4f}, "
            f"Noise: {self.likelihood.noise.item(): .6f}"
        )
