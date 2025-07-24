import torch
import torch.nn as nn
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import (
    ScaleKernel,
    RBFKernel,
    MaternKernel,
    PeriodicKernel,
    LinearKernel,
    PolynomialKernel,
    SpectralMixtureKernel,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import GammaPrior


def build_gp(train_set, val_set, gp_type: str = "vanilla", **kwargs):
    """function to build a Gaussian Process model based on the specified type."""

    # check the type of GP model
    supported_gp_types = [
        "vanilla",
        "deep_kernel",
        "enhanced_deep_kernel",
        "enhanced",
        "hierarchical",
        "mini_batch_vanilla",
    ]
    assert gp_type in supported_gp_types, (
        f"Unsupported GP type: {gp_type}. "
        f"Supported types are: {', '.join(supported_gp_types)}."
    )
    print(f"Building {gp_type} GP model...")

    if gp_type == "vanilla":
        return VanillaGP(
            train_set, val_set, device=kwargs.get("device", torch.device("cpu"))
        )
    elif gp_type == "mini_batch_vanilla":
        return MiniBatchVanillaGP(
            train_set,
            val_set,
            device=kwargs.get("device", torch.device("cpu")),
            batch_size=kwargs.get("batch_size", 1000),
        )
    elif gp_type == "deep_kernel":
        return DeepKernelGP(
            train_set,
            val_set,
            device=kwargs.get("device", torch.device("cpu")),
            hidden_dim=kwargs.get("hidden_dim", 64),
        )
    elif gp_type == "enhanced_deep_kernel":
        return EnhancedDeepKernelGP(
            train_set,
            val_set,
            device=kwargs.get("device", torch.device("cpu")),
            hidden_dims=kwargs.get("hidden_dims", [64, 128, 64]),
            feature_dim=kwargs.get("feature_dim", 32),
            mean_type=kwargs.get("mean_type", "linear"),
            kernel_type=kwargs.get("kernel_type", "composite"),
        )
    elif gp_type == "enhanced":
        return EnhancedGPModel(
            train_set, val_set, device=kwargs.get("device", torch.device("cpu"))
        )
    elif gp_type == "hierarchical":
        return HierarchicalGPModel(
            train_set, val_set, device=kwargs.get("device", torch.device("cpu"))
        )
    else:
        raise ValueError(f"Unknown GP type: {gp_type}.")


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


class MiniBatchVanillaGP(ExactGP):
    """Vanilla GP with mini-batch training to reduce memory usage."""

    def __init__(self, train_set, val_set, device=None, batch_size=1000):
        self.full_train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.device = device

        # Initialize with a subset for the ExactGP parent class
        train_x, train_y = [t.to(device) for t in train_set.tensors]
        assert (
            len(train_x) >= batch_size
        ), f"Training set must have at least {batch_size} samples."
        subset_idx = torch.randperm(len(train_x))[:batch_size]
        self.current_train_x = train_x[subset_idx]
        self.current_train_y = train_y[subset_idx].squeeze()

        self.val_x, self.val_y = [t.to(device) for t in val_set.tensors]

        likelihood = GaussianLikelihood()
        super().__init__(self.current_train_x, self.current_train_y, likelihood)

        self.likelihood = likelihood.to(device)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.to(device)

    def update_training_data(self):
        """Randomly sample a new batch of training data."""
        train_x, train_y = [t.to(self.device) for t in self.full_train_set.tensors]
        n_samples = len(train_x)

        if n_samples <= self.batch_size:
            subset_idx = torch.arange(n_samples)
        else:
            subset_idx = torch.randperm(n_samples)[: self.batch_size]

        self.current_train_x = train_x[subset_idx]
        self.current_train_y = train_y[subset_idx].squeeze()

        # Update the parent class training data
        self.set_train_data(self.current_train_x, self.current_train_y, strict=False)

    def _compute_loss(self, mll, x, y):
        assert y.ndim == 1, "Targets should be 1D."
        output = self.forward(x)
        return -mll(output, y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def train_step(self, mll, optimizer):
        self.train()
        self.likelihood.train()

        # Update to a new batch
        self.update_training_data()

        optimizer.zero_grad()
        loss = self._compute_loss(mll, self.current_train_x, self.current_train_y)
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


class EnhancedDeepKernelGP(ExactGP):
    def __init__(
        self,
        train_set,
        val_set,
        device=None,
        hidden_dims=[64, 128, 64],
        feature_dim=32,
        mean_type="linear",
        kernel_type="composite",
    ):
        self.train_set = train_set
        self.val_set = val_set
        self.train_x, self.train_y = [t.to(device) for t in train_set.tensors]
        self.val_x, self.val_y = [t.to(device) for t in val_set.tensors]

        # likelihood
        likelihood = GaussianLikelihood()

        # Initialize ExactGP model
        super().__init__(self.train_x, self.train_y.squeeze(), likelihood)
        self.likelihood = likelihood.to(device)

        # Model parameters
        self.input_dim = self.train_x.shape[1]
        self.hidden_dims = hidden_dims
        self.feature_dim = feature_dim

        # build feature extractor
        self.feature_extractor = self._build_feature_extractor()

        # build mean function
        self.mean_module = self._build_mean_function(mean_type)

        # build covariance module
        self.covar_module = self._build_covariance_module(kernel_type)

        # Move to device
        self.to(device)

    def _build_feature_extractor(self):
        """build the feature extractor fully connected network"""
        layers = []

        # Input layer
        prev_dim = self.input_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            # Add a linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # batch normalization
            layers.append(nn.BatchNorm1d(hidden_dim))
            # ReLU activation
            layers.append(nn.ReLU())
            # Dropout layer
            layers.append(nn.Dropout(p=0.1))
            # Update previous dimension
            prev_dim = hidden_dim

        # final projection layer
        layers.append(nn.Linear(prev_dim, self.feature_dim))

        return nn.Sequential(*layers)

    def _build_mean_function(self, mean_type):
        """build the mean function based on the specified type"""
        if mean_type == "constant":
            return ConstantMean()
        elif mean_type == "linear":
            return LinearMean(input_size=self.feature_dim)
        else:
            raise ValueError(f"Unknown mean type: {mean_type}")

    def _build_covariance_module(self, kernel_type):
        """build the covariance module based on the specified type"""
        if kernel_type == "rbf":
            return ScaleKernel(RBFKernel(ard_num_dims=self.feature_dim))

        elif kernel_type == "matern":
            return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=self.feature_dim))

        elif kernel_type == "spectral":
            return ScaleKernel(
                SpectralMixtureKernel(num_mixtures=4, ard_num_dims=self.feature_dim)
            )

        elif kernel_type == "composite":
            # Multi-scale RBF kernels
            rbf_short = ScaleKernel(
                RBFKernel(
                    ard_num_dims=self.feature_dim,
                    lengthscale_prior=GammaPrior(0.5, 0.2),
                )
            )

            rbf_medium = ScaleKernel(
                RBFKernel(
                    ard_num_dims=self.feature_dim,
                    lengthscale_prior=GammaPrior(2.0, 0.2),
                )
            )

            rbf_long = ScaleKernel(
                RBFKernel(
                    ard_num_dims=self.feature_dim,
                    lengthscale_prior=GammaPrior(5.0, 0.2),
                )
            )
            # NOTE: Matern and Linear kernels run into numerical stability issues.
            # # Matern kernel for less smooth patterns
            # matern_kernel = ScaleKernel(
            #     MaternKernel(
            #         nu=1.5,
            #         ard_num_dims=self.feature_dim,
            #         lengthscale_prior=GammaPrior(1.0, 0.2),
            #     )
            # )

            # # Linear kernel for trends
            # linear_kernel = ScaleKernel(LinearKernel())

            # Combine kernels
            return rbf_short + rbf_medium + rbf_long

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


class EnhancedGPModel(ExactGP):
    """Enhanced Gaussian Process model with composite kernel for complex patterns ."""

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

        # Mean function
        self.mean_module = LinearMean(input_size=self.train_x.shape[1])

        # Composite kernel
        # 1. Smooth long-term trends
        smooth_kernel = ScaleKernel(RBFKernel(lengthscale_prior=GammaPrior(2.0, 0.15)))

        # 2. Local variations with Matern kernel
        local_kernel = ScaleKernel(
            MaternKernel(nu=2.5, lengthscale_prior=GammaPrior(1.0, 0.1))
        )

        # 3. Periodic patterns
        periodic_kernel = ScaleKernel(
            PeriodicKernel(
                lengthscale_prior=GammaPrior(1.0, 0.1),
                period_length_prior=GammaPrior(2.0, 0.15),
            )
        )

        # 4. Linear trends
        linear_kernel = ScaleKernel(LinearKernel())

        # Combine kernels additively
        self.covar_module = (
            smooth_kernel + local_kernel + periodic_kernel + linear_kernel
        )

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


class HierarchicalGPModel(ExactGP):
    """Hierarchical Gaussian Process model for multi-scale patterns."""

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

        # Mean function
        self.mean_module = LinearMean(input_size=self.train_x.shape[1])

        # multiple RBF kernels for different scales

        # global kernel for long-term trends
        global_kernel = ScaleKernel(RBFKernel(lengthscale_prior=GammaPrior(2.0, 0.15)))

        # medium scale kernel
        medium_kernel = ScaleKernel(RBFKernel(lengthscale_prior=GammaPrior(1.0, 0.1)))

        # small scale  kernel
        local_kernel = ScaleKernel(RBFKernel(lengthscale_prior=GammaPrior(0.3, 0.5)))

        # polynomial kernel for local variations
        poly_kernel = ScaleKernel(PolynomialKernel(power=2))

        # combine all scales
        self.covar_module = global_kernel + medium_kernel + local_kernel + poly_kernel

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
