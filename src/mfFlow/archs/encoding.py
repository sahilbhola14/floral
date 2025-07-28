import math
import torch
import torch.nn as nn
from mfFlow.archs import MLP


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
        else:
            self.register_buffer("log_bandwidth", torch.tensor(math.log(10.0)))

        # Keep similar network sizes to original but fix output dimension
        self.gamma_net = MLP(
            in_dim=self.num_centers,
            width=[64, 64],  # Same as original
            out_dim=self.latent_dim,  # Match latent_dim for proper FiLM
            activations=[nn.ReLU(), nn.ReLU(), None],
        )

        self.beta_net = MLP(
            in_dim=self.num_centers,
            width=[64, 64],  # Same as original
            out_dim=self.latent_dim,  # FIX: Match latent_dim, not 1
            activations=[nn.ReLU(), nn.ReLU(), None],
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
        distances = torch.norm(delta, dim=-1)

        # RBF with learnable bandwidth
        bandwidth = torch.exp(self.log_bandwidth)
        return torch.exp(-bandwidth * distances**2)

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
        beta = self.beta_net(mod_features)  # (batch_size, latent_dim) - FIXED

        # Apply FiLM modulation properly
        # Original: torch.sum(x * gamma, dim=-1, keepdim=True) + beta  # WRONG!
        # Fixed: element-wise modulation
        modulated = x * (1 + gamma) + beta  # (batch_size, latent_dim)

        # Project to output dimension - THIS WAS MISSING
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
    def __init__(self, in_channels, embed_dim, num_heads=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_proj = self.proj(x)  # (B, embed_dim, H, W)
        x_flat = x_proj.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = self.norm(attn_out)
        return attn_out.mean(dim=1)  # shape: (B, embed_dim)


class FiLM(nn.Module):
    """Vanilla Feature Layer Modulation (FiLM) layer

    Give an embedding vector of size (batch_size, emb_dim), this layer modulates a
    tensor of size (batch_size, num_channels, H, W) by scaling and shifting it.

    """

    def __init__(self, emb_dim: int, num_channels: int):
        """
        Args:
            emb_dim (int): Dimension of the embedding vector
            num_channels (int): Number of channels in the output tensor, which is
                                modulated.
        """
        super(FiLM, self).__init__()
        self.emb_dim = emb_dim
        self.num_channels = num_channels
        self.scale = nn.Linear(self.emb_dim, self.num_channels)
        self.shift = nn.Linear(self.emb_dim, self.num_channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        """forward pass of the FiLM layer
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, H, W)
                            that will be modulated.
            emb (torch.Tensor): Embedding tensor of shape (batch_size, emb_dim) that
                                will be used to modulate the input tensor.
        """
        assert (
            emb.shape[-1] == self.emb_dim
        ), f"Embedding dimension mismatch: {emb.shape[-1]} != {self.emb_dim}"
        assert emb.ndim == 2, "Embedding tensor must be 2D (batch_size, emb_dim)"
        assert (
            x.shape[1] == self.num_channels
        ), f"Number of channels mismatch: {x.shape[1]} != {self.num_channels}"
        assert x.ndim == 4, "Input tensor must be 4D (batch_size, num_channels, H, W)"
        scale = self.scale(emb).view(-1, self.num_channels, 1, 1)
        shift = self.shift(emb).view(-1, self.num_channels, 1, 1)
        return x * (1 + scale) + shift
