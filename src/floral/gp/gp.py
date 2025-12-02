# src/floral/gp/gp.py
"""
Gaussian process modules
GPPrior adapted from:
https://github.com/GavinKerrigan/functional_flow_matching/blob/master/util/gaussian_process.py
"""
import torch
import math
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal


def get_gp_prior(
    lengthscale: float,
    outputscale: float,
    confidence: float = 0.0,
):
    """build a gp prior"""
    # check keys
    return GPPrior(
        lengthscale=lengthscale, outputscale=outputscale, confidence=confidence
    )


class GPPrior(ExactGP):
    """GP Prior for base measure"""

    def __init__(
        self,
        kernel=None,
        mean=None,
        lengthscale=None,
        outputscale=None,
        confidence=1.0,
        **kwargs,
    ):
        likelihood = GaussianLikelihood()
        super(GPPrior, self).__init__(
            train_inputs=None, train_targets=None, likelihood=likelihood
        )
        self.confidence = confidence
        self.likelihood = likelihood
        self.mean_module = self._build_mean_module(mean=mean)
        self.covar_module = self._build_covariance_module(
            kernel=kernel,
            lengthscale=lengthscale,
            outputscale=outputscale,
            confidence=self.confidence,
        )
        # set to eval mode
        self.eval()

        # freeze mean
        for p in self.mean_module.parameters():
            p.requires_grad = False

        # freeze covariance kernel (outputscale + lengthscale)
        for p in self.covar_module.parameters():
            p.requires_grad = False

        #  freeze likelihood (prior sampling does not required likelihood)
        for p in self.likelihood.parameters():
            p.requires_grad = False

    def _build_mean_module(self, mean=None):
        """build the mean module
        Args:
            mean:
                mean function
        """
        return ConstantMean() if mean is None else mean

    def _confidence_to_scale(
        self, confidence, min_factor: float = 0.00001, max_factor: float = 1.0
    ):
        confidence = torch.clamp(torch.tensor(confidence), 0.0, 1.0)
        log_min = math.log(min_factor)
        log_max = math.log(max_factor)
        inter = (1.0 - confidence) * log_max + confidence * log_min
        return torch.exp(inter)

    def _build_covariance_module(
        self, kernel=None, lengthscale=None, outputscale=None, confidence=1.0
    ):
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
                scale = self._confidence_to_scale(confidence=1.0)
                self.outputscale = torch.FloatTensor([0.1])
            else:
                assert (
                    isinstance(outputscale, float) and outputscale > 0
                ), "outputscale must be positive float"
                scale = self._confidence_to_scale(confidence=confidence)
                self.outputscale = torch.FloatTensor([outputscale])
            self.outputscale = scale * self.outputscale
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
    def sample(
        self,
        domain: torch.Tensor,
        batch_size: int,
        field_channels: int,
        field_dims: list,
    ):
        """sample functions from the GP prior
        Args:
            domain (torch.Tensor):
                input domain (flattened)
            batch_size (int):
                number of batches to sample
            field_channels (int):
                number of output channels (for multi-output GP)
            field_dims (list):
                shape of the field
        Returns:
            samples (torch.Tensor): sampled functions of shape
            (batch_size, n_channels, *dims)
        Note:
            1. For multiple channels, we assume independence between channels.
            That is, i.i.d. draws from the same GP prior.

        """
        # get the GP distribution
        distribution = self.forward(domain)
        # generate samples
        samples = distribution.sample(torch.Size([batch_size * field_channels]))
        samples = samples.view(batch_size, field_channels, *field_dims)
        return samples
