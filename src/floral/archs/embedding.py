# src/floral/archs/embedding.py
"""
contains different types of embedding that can be used to modulate data.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class MLP(nn.Module):
    """Multi-layer perceptron with flexible architecture"""

    def __init__(self, in_dim, width, out_dim, activations, dropout=0.0, norm=None):
        super().__init__()
        layers = []
        dims = [in_dim] + width + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Add normalization if specified
            if norm == "layer" and i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
            elif norm == "batch" and i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))

            # Add activation
            if i < len(activations) and activations[i] is not None:
                layers.append(activations[i])

            # Add dropout
            if dropout > 0 and i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RBFFiLM(nn.Module):
    """
    RBF-FiLM
    """

    def __init__(
        self,
        latent_dim: int,
        num_centers: int,
        nd: int,
        nx: int,
        learnable_bandwidth=True,
        improved_centers=True,
        **kwargs,
    ):
        """
        Args:
            latent_dim (int): Dimension of the latent space
            num_centers (int): Number of RBF centers
            nd (int): Number of domain dimensions
            nx (int): Output dimension (should match state dimension)
            learnable_bandwidth (bool): Whether to learn bandwidth
            improved_centers (bool): Whether to use better center initialization
        """
        super().__init__()
        self.num_centers = num_centers
        self.latent_dim = latent_dim
        self.nd = nd
        self.nx = nx
        self.dropout = kwargs.get("dropout", 0.0)

        # Initialize centers with better coverage if requested
        if improved_centers:
            self.centers = self._initialize_better_centers()
        else:
            # Original initialization
            self.centers = nn.Parameter(
                torch.stack(
                    [torch.linspace(0, 1, self.num_centers) for _ in range(self.nd)],
                    dim=-1,
                ),
                requires_grad=True,
            )

        # Learnable bandwidth (single value for simplicity)
        if learnable_bandwidth:
            self.log_bandwidth = nn.Parameter(torch.tensor(math.log(10.0)))
            self.log_bandwidth = nn.Parameter(
                torch.ones(self.num_centers) * math.log(10.0)
            )
        else:
            self.register_buffer(
                "log_bandwidth", torch.ones(self.num_centers) * math.log(10.0)
            )

        # Keep similar network sizes to original but fix output dimension
        self.gamma_net = MLP(
            in_dim=self.num_centers,
            width=[64, 64],  # Same as original
            out_dim=self.latent_dim,  # Match latent_dim for proper FiLM
            activations=[nn.ReLU(), nn.ReLU(), None],
            dropout=self.dropout,
        )

        self.beta_net = MLP(
            in_dim=self.num_centers,
            width=[64, 64],  # Same as original
            out_dim=self.latent_dim,  # FIX: Match latent_dim, not 1
            activations=[nn.ReLU(), nn.ReLU(), None],
            dropout=self.dropout,
        )

        # Output projection to match state dimension
        # This is the key missing piece in your original
        self.output_proj = nn.Linear(latent_dim, nx)

    def _initialize_better_centers(self):
        """Better center initialization without much complexity"""
        if self.nd == 1:
            # Chebyshev nodes for better approximation
            i = torch.arange(1, self.num_centers + 1).float()
            centers = 0.5 * (
                1 + torch.cos((2 * i - 1) * math.pi / (2 * self.num_centers))
            )
            centers = centers.unsqueeze(1)
        elif self.nd == 2:
            # Simple grid initialization
            side = int(math.ceil(math.sqrt(self.num_centers)))
            x = torch.linspace(0, 1, side)
            y = torch.linspace(0, 1, side)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            centers = torch.stack([xx.flatten(), yy.flatten()], dim=1)[
                : self.num_centers
            ]
        else:
            # Random for higher dimensions
            centers = torch.rand(self.num_centers, self.nd)

        assert centers.shape == (self.num_centers, self.nd)

        return nn.Parameter(centers, requires_grad=True)

    def _rbf_encoding(self, mod):
        """Simple RBF encoding - minimal change from original"""
        # Ensure mod is properly bounded
        mod = torch.clamp(mod, 0, 1)

        # Compute distances
        delta = mod.unsqueeze(1) - self.centers.unsqueeze(0)
        sq_dist = (delta * delta).sum(-1)

        # RBF with learnable bandwidth
        bandwidth = self.log_bandwidth.exp().unsqueeze(0)  # (1, num_centers)
        return torch.exp(-bandwidth * sq_dist)

    def forward(self, x: torch.Tensor, mod: torch.Tensor):
        """
        Forward pass - key fix is proper dimensionality handling

        Args:
            x (torch.Tensor): Input features of shape (batch_size, latent_dim)
            mod (torch.Tensor): Domain coordinates of shape (batch_size, nd)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, nx)
        """
        # Get RBF features (same as original)
        mod_features = self._rbf_encoding(mod)

        # Generate FiLM parameters
        gamma = self.gamma_net(mod_features)  # (batch_size, latent_dim)
        beta = self.beta_net(mod_features)  # (batch_size, latent_dim)

        # Apply FiLM
        modulated = x * (1 + gamma) + beta  # (batch_size, latent_dim)

        output = self.output_proj(modulated)  # (batch_size, nx)

        return output


class RBFFiLMAttention(nn.Module):
    """Radial Basis Function FiLM (Good for encodding spatial information)"""

    def __init__(
        self,
        latent_dim: int,
        num_centers: int,
        nd: int,
        nx: int,
        kernel_type="gaussian",
        learnable_bandwidth=True,
        multi_scale=True,
        use_attention=False,
    ):
        """
        Args:
        latent_dim (int): Dimension of the latent space
        num_centers (int): Number of RBF centers per dimension
        nd (int): Number of domain dimensions
        nx (int): Output dimension (should match state dimension)
        kernel_type (str): Type of RBF kernel
        learnable_bandwidth (bool): Whether to learn bandwidth parameters
        multi_scale (bool): Whether to use multiple bandwidth scales
        use_attention (bool): Whether to use attention over RBF features
        """
        super(RBFFiLMAttention, self).__init__()

        self.num_centers = num_centers
        self.latent_dim = latent_dim
        self.nd = nd
        self.nx = nx
        self.kernel_type = kernel_type
        self.learnable_bandwidth = learnable_bandwidth
        self.multi_scale = multi_scale
        self.use_attention = use_attention

        assert self.use_attention is False, "Under construction"

        # intialize the learnable RBF centers
        self.centers = self._initialize_centers()  # (num_centers, nd)

        self.num_scales = 4 if self.multi_scale else 1

        # learnable bandwidth parameters (num_centers, num_scales)
        if learnable_bandwidth:
            if multi_scale:
                # Multiple bandwidth scales for multi-resolution features
                self.bandwidths = nn.Parameter(
                    torch.linspace(-2, 2, self.num_scales)
                    .unsqueeze(0)
                    .repeat(self.num_centers, 1)
                    .exp()
                )  # Shape: [num_centers, num_scales]
            else:
                self.bandwidths = nn.Parameter(torch.zeros(self.num_centers).exp())
        else:
            if multi_scale:
                self.register_buffer(
                    "bandwidths",
                    torch.ones(self.num_centers, self.num_scales)
                    * math.log(10.0).exp(),
                )
            else:
                self.register_buffer(
                    "bandwidths", torch.ones(self.num_centers) * math.log(10.0).exp()
                )

        # Calculate total RBF feature dimension
        self.total_rbf_dim = self.num_centers * self.num_scales

        # FiLM gamma net
        self.gamma_net = MLP(
            in_dim=self.total_rbf_dim,
            width=[128, 128, 64],
            out_dim=self.latent_dim,
            activations=[nn.SiLU(), nn.SiLU(), nn.SiLU(), None],
            dropout=0.1,
            norm="layer",
        )

        # FiLM beta net
        self.beta_net = MLP(
            in_dim=self.total_rbf_dim,
            width=[128, 128, 64],
            out_dim=self.latent_dim,
            activations=[nn.SiLU(), nn.SiLU(), nn.SiLU(), None],
            dropout=0.1,
            norm="layer",
        )

        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim * 2),
            nn.LayerNorm(self.latent_dim * 2),
            nn.SiLU(),
            nn.Linear(self.latent_dim * 2, self.nx),
        )

    def _initialize_centers(self):
        """Initialize the centers"""
        if self.nd == 1:
            # 1D: Use Chebyshev nodes for better approximation
            i = torch.arange(1, self.num_centers + 1).float()
            centers = 0.5 * (
                1 + torch.cos((2 * i - 1) * math.pi / (2 * self.num_centers))
            )
            centers = centers.unsqueeze(1)
        elif self.nd == 2:
            # 2D: Use grid with some jitter for better coverage
            side_length = int(math.ceil(math.sqrt(self.num_centers)))
            x = torch.linspace(0, 1, side_length)
            y = torch.linspace(0, 1, side_length)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            centers = torch.stack([xx.flatten(), yy.flatten()], dim=1)
            # Add small random jitter
            centers += 0.05 * torch.randn_like(centers)
            centers = torch.clamp(centers, 0, 1)
            # Take only num_centers points
            centers = centers[: self.num_centers]
        else:
            # Higher dimensions: Use random initialization with Latin hypercube sampling
            centers = torch.rand(self.num_centers, self.nd)
        assert centers.shape == (self.num_centers, self.nd)
        return nn.Parameter(centers, requires_grad=True)

    def _rbf_kernel(self, distances, bandwidth):
        """Compute RBF kernel values"""
        if self.kernel_type == "gaussian":
            return torch.exp(-0.5 * distances**2 / bandwidth**2)
        else:
            raise NotImplementedError("Under construction")

    def _rbf_encoding(self, mod):
        """encding for the mod using radial basis functions
        TODO:
            - Make input  for any range.
            Can do some normalization to make it between [0, 1]
        """
        assert mod.min() >= 0 and mod.max() <= 1
        batch_size = mod.shape[0]
        # compute the distances
        delta = mod.unsqueeze(1) - self.centers.unsqueeze(
            0
        )  # (batch_size, num_center, nd)
        distances = torch.sqrt(torch.sum(delta**2, dim=-1))

        # bandwidth

        if self.multi_scale:
            # Expand dimensions for broadcasting
            distances_expanded = distances.unsqueeze(-1)  # (batch_size, num_centers, 1)
            bandwidths_expanded = self.bandwidths.unsqueeze(0).to(
                mod.device
            )  # (1, num_centers, num_scales)
            # compute the RBF values
            rbf_values = self._rbf_kernel(distances_expanded, bandwidths_expanded)

            # flatten to (batch_size, num_centers * num_scales)
            rbf_features = rbf_values.view(batch_size, -1)
        else:
            rbf_features = self._rbf_kernel(distances, self.bandwidths.unsqueeze(0))

        assert rbf_features.shape == (batch_size, self.total_rbf_dim)

        return rbf_features

    def forward(self, x: torch.Tensor, mod: torch.Tensor):
        """forward"""
        batch_size = x.shape[0]
        assert x.ndim == 2 and x.shape == (batch_size, self.latent_dim)
        assert mod.ndim == 2 and mod.shape == (batch_size, self.nd)
        # get the rbf features
        rbf_features = self._rbf_encoding(mod)  # (batch_size, num_center * num_scales)

        # Generate FiLM parameters
        gamma = self.gamma_net(rbf_features)  # (batch_size, latent_dim)
        beta = self.beta_net(rbf_features)  # (batch_size, latent_dim)

        # Apply FiLM modulation (proper element-wise modulation)
        modulated = x * (1 + gamma) + beta  # (batch_size, latent_dim)

        # Project to output dimension with residual connection
        output = self.output_proj(modulated)  # (batch_size, nx)
        return output


class SpatialAttentionPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        embed_dim: int = 128,
        num_attention_heads=1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads
        self.conv = conv_nd(
            dims=2, in_channels=in_channels, out_channels=self.embed_dim, kernel_size=1
        )
        self.q = nn.Parameter(torch.randn(1, self.embed_dim))  # learnable query
        self.proj = MLP(
            in_dim=self.embed_dim,
            out_dim=self.latent_dim,
            width=[32],
            activations=[nn.SiLU(), None],
            dropout=self.dropout,
            norm="layer",
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_attention_heads,
            batch_first=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)  # (B, embed_dim, H, W)
        x = x.view(B, self.embed_dim, H * W).transpose(1, 2)  # (B, HW, embed_dim)
        q = self.q.expand(B, -1).unsqueeze(1)  # (B, 1, embed_dim)
        out, _ = self.attn(q, x, x)  # (B, 1, embed_dim)
        return self.proj(out.squeeze(1))  # (B, latent_dim)


class SpatialAdaptivePooling(nn.Module):
    def __init__(
        self, in_channels: int, latent_dim: int, dropout: float = 0.1, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.pool_size = kwargs.get("pool_size", 4)

        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.proj = MLP(
            in_dim=self.in_channels * self.pool_size**2,
            out_dim=self.latent_dim,
            width=[32],
            activations=[nn.SiLU(), None],
            dropout=self.dropout,
            norm="layer",
        )

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.pool(x)  # (B, C, 4, 4)
        return self.proj(out.flatten(1))  # (B, latent_dim)


class SpatialFiLM(nn.Module):
    """Give a mod of shape (B, ch_mod, *grid) and target of shape
    (B, ch_target, *grid), apply the modulation across each dim
    (with same across channels).
    Args:
        mod_ch (int):
            Number of channels in the modulating tensor
        target_ch (int):
            Number of chanenls in the target tensor
    """

    def __init__(
        self,
        mod_ch: int,
        target_ch: int,
        **kwargs,
    ):
        super(SpatialFiLM, self).__init__()
        self.mod_ch = mod_ch
        self.target_ch = target_ch

        width = kwargs.get("width", [64, 64])
        dropout = kwargs.get("dropout", 0.0)
        activation = nn.SiLU()

        # film layer
        self.film_net = MLP(
            in_dim=self.mod_ch,
            out_dim=2 * self.target_ch,
            width=width,
            activations=[activation for _ in range(len(width))] + [None],
            dropout=dropout,
        )

    def _check_input(self, target: torch.Tensor, mod: torch.Tensor):
        """check the input"""
        _, target_ch, *target_grid = target.shape
        _, mod_ch, *mod_grid = mod.shape

        assert target_ch == self.target_ch, "incorrect target channels."
        assert mod_ch == self.mod_ch, "incorrect mod channels."

        assert all([ii == jj for ii, jj in zip(target_grid, mod_grid)]), "invalid input"

    def forward(self, target: torch.Tensor, mod: torch.Tensor):
        # check input
        self._check_input(target, mod)

        _, mod_ch, *mod_grid = mod.shape

        mod_flat = mod.flatten(start_dim=2).transpose(1, 2)  # (B, ..., mod_ch)

        gamma, beta = torch.chunk(self.film_net(mod_flat), chunks=2, dim=-1)
        gamma, beta = gamma.transpose(1, 2), beta.transpose(1, 2)  # (B, mod_ch, ...)
        gamma = gamma.view(-1, mod_ch, *mod_grid)  # (B, mod_ch, *mod_grid)
        beta = beta.view(-1, mod_ch, *mod_grid)  # (B, mod_ch, *mod_grid)
        return target * (1 + gamma) + beta  # (B, ch_target, dim_target)


class AttentionPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn_proj = nn.Linear(in_channels, 1)

    def forward(self, x):
        # x: (B, C, *grid) -> flatten
        B, C, *grid = x.shape
        N = int(torch.tensor(grid).prod())
        x_flat = x.view(B, C, N).transpose(1, 2)  # (B, N, C)

        # scores -> (B, N, 1)
        scores = self.attn_proj(x_flat)
        weights = F.softmax(scores, dim=1)  # (B, N, 1)

        # weighted sum -> (B, C)
        pooled = torch.sum(weights * x_flat, dim=1)
        return pooled
