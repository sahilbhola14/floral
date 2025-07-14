import torch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal


class GPRegressionModel(ExactGP):
    """Gaussian Process model for regression tasks."""

    def __init__(self, train_set, val_set):

        self.train_set = train_set
        self.val_set = val_set
        # Extract training and validation data
        self.train_x, self.train_y = train_set.tensors
        self.val_x, self.val_y = val_set.tensors
        # Gaussian Likelihood
        likelihood = GaussianLikelihood()
        # Initialize the GP model with a constant mean and RBF kernel
        super(GPRegressionModel, self).__init__(self.train_x, self.train_y, likelihood)
        # Set the likelihood
        self.likelihood = likelihood
        # Mean model
        self.mean = ConstantMean()
        # Covariance function
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        """Forward pass through the GP model."""
        mean_x = self.mean(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def _compute_loss(self, mll, in_features, targets):
        assert targets.ndim == 1, "Targets should be a 1D tensor."
        # forward pass
        output = self.forward(in_features)
        # loss
        loss = -mll(output, targets)
        return loss

    def train_step(self, mll):
        """Training step"""
        # set the model to training mode
        self.train()
        self.likelihood.train()
        # compute the loss
        loss = self._compute_loss(mll, self.train_x, self.train_y.view(-1))

        return loss

    @torch.no_grad()
    def val_step(self, mll):
        """Validation step"""
        # set the model to evaluation mode
        self.eval()
        self.likelihood.eval()
        # compute the loss
        loss = self._compute_loss(mll, self.val_x, self.val_y.view(-1))
        return loss
