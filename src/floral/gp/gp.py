# src/floral/gp/gp.py
"""
Gaussian process modules
GPPrior adapted from:
https://github.com/GavinKerrigan/functional_flow_matching/blob/master/util/gaussian_process.py
"""
import torch

# import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from floral.utils import check_keys


def get_gp_prior(prior_config):
    """build a gp prior"""
    # check keys
    required_keys = ["lengthscale", "outputscale"]
    check_keys(prior_config, required_keys)
    return GPPrior(**prior_config)


class GPPrior(ExactGP):
    """GP Prior for base measure"""

    def __init__(
        self, kernel=None, mean=None, lengthscale=None, outputscale=None, **kwargs
    ):
        likelihood = GaussianLikelihood()
        super(GPPrior, self).__init__(
            train_inputs=None, train_targets=None, likelihood=likelihood
        )
        self.likelihood = likelihood
        self.mean_module = self._build_mean_module(mean=mean)
        self.covar_module = self._build_covariance_module(
            kernel=kernel, lengthscale=lengthscale, outputscale=outputscale
        )
        # set to eval mode
        self.eval()

    def _build_mean_module(self, mean=None):
        """build the mean module
        Args:
            mean:
                mean function
        """
        return ConstantMean() if mean is None else mean

    def _build_covariance_module(self, kernel=None, lengthscale=None, outputscale=None):
        """build the covariance module"""
        eps = 1e-10  # jitter
        nu = 0.5  # smoothness
        if kernel is None:
            # set length scale
            if lengthscale is None:
                self.lengthscale = torch.FloatTensor([0.01])
            else:
                assert (
                    isinstance(lengthscale, float) and lengthscale > 0
                ), "lengthscale must be positive float"
                self.lengthscale = torch.FloatTensor([lengthscale])
            # set output scale
            if outputscale is None:
                self.outputscale = torch.FloatTensor([0.1])
            else:
                assert (
                    isinstance(outputscale, float) and outputscale > 0
                ), "outputscale must be positive float"
                self.outputscale = torch.FloatTensor([outputscale])
            # build Matern kernel
            base_kernel = MaternKernel(nu=nu, eps=eps)
            base_kernel.lengthscale = self.lengthscale
            # scale kernel
            covar_kernel = ScaleKernel(base_kernel)
            covar_kernel.outputscale = self.outputscale
        else:
            covar_kernel = kernel

        return covar_kernel

    def forward(self, x):
        """forward pass"""
        assert x.ndim == 2 and isinstance(
            x, torch.Tensor
        ), "Input should be a 2D tensor."
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @torch.no_grad()
    def sample(self, x, dims, n_samples: int = 1, n_channels: int = 1):
        """sample functions from the GP prior
        Args:
            x (torch.Tensor):
                Flattened input domain
            n_samples (int):
                number of function samples to draw
            n_channels (int):
                number of output channels (for multi-output GP)
        Returns:
            samples (torch.Tensor): sampled functions of shape
            (n_samples, n_channels, *grid)
        Note:
            1. For multiple channels, we assume independence between channels.
            That is, i.i.d. draws from the same GP prior.

        """
        assert x.ndim == 2 and isinstance(
            x, torch.Tensor
        ), "Input should be a 2D tensor."
        assert x.shape[1] == len(
            dims
        ), "Input feature dimension must match the length of dims."

        distribution = self.forward(x)  # get the GP distribution
        samples = distribution.sample(
            torch.Size([n_samples * n_channels])
        )  # (n_samples * n_channels, N)
        samples = samples.view(
            n_samples, n_channels, *dims
        )  # (n_samples, n_channels, *dims)
        assert samples.shape == (n_samples, n_channels, *dims), "incorrect samples"
        return samples
