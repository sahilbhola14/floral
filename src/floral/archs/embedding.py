# src/floral/archs/embedding.py
"""
Useful embeddings
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_domain_encoder(pos_encoder: str = "fourier", **kwargs: dict):
    """
    get encoder for domain.
    This is done to enrich the feature space.
    Args:
        pos_encoder (str):
            type of position encoder
    Returns:
        encoder (nn.Module):
            position encoder module
        output_features (int):
            dimensionality of the embedding
    """
    available_pos_encoders = ["rbf", "fourier"]

    if pos_encoder == "rbf":
        encoder = RBFEncoding(
            ndim=kwargs.get("ndim"),
            n_centers=kwargs.get("n_centers", 10),
            learnable_centers=kwargs.get("learnable_centers", True),
            domain_min=kwargs.get("domain_min", 0.0),
            domain_max=kwargs.get("domain_max", 1.0),
        )
    elif pos_encoder == "fourier":
        encoder = FourierEncoding(
            ndim=kwargs.get("ndim"),
            n_fourier_modes=kwargs.get("n_fourier_modes", 5),
            learnable_modes=kwargs.get("learnable_modes", True),
        )
    else:
        raise ValueError(
            f"Invalid pos_encoder: {pos_encoder}, "
            f"choose from {', '.join(available_pos_encoders)}"
        )

    assert hasattr(encoder, "output_features"), "need to specify the output features"

    return encoder, encoder.output_features


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


class SpatialAttentivePooling(nn.Module):
    """
    Spatial Attentive Pooling to convert (batch_size, in_channels, *dims) to
    (batch_size, in_channels) irrespective of *dims. This is performed to
    make sure there is a (learnable) weighted averaging, rather than simple
    mean.
    """

    def __init__(self, in_channels: int, **kwargs):
        super(SpatialAttentivePooling, self).__init__()
        self.in_channels = in_channels
        activation = nn.SiLU()
        self.net = MLP(
            in_dim=self.in_channels,
            out_dim=self.in_channels,
            activations=[activation, activation, None],
            width=[32, 32],
            dropout=kwargs.get("dropout", 0.0),
        )

    def forward(self, target: torch.Tensor):
        """forwar pass"""
        batch_size, channels, *dims = target.shape
        assert channels == self.in_channels
        # reshape target
        reshape_target = target.flatten(start_dim=2).transpose(1, 2)
        # compute weights
        weights = self.net(reshape_target)
        weights = F.softmax(weights, dim=1)
        # Weighted pooling (elementwise multiply and sum)
        pooled = torch.sum(weights * reshape_target, dim=1)  # (B, C)

        return pooled


class ChannelFiLM(nn.Module):
    """
    Class for Spatial FiLM.
    Modulate a target tensor of shape (batch_size, target_channels, *dims) as
    target = target * (1 + gamma) + beta), where gamma and beta are computed using
    embed_features. This class modulates each channels with different weights.
    """

    def __init__(self, target_channels: int, embed_features: int, **kwargs):
        super(ChannelFiLM, self).__init__()
        self.target_channels = target_channels
        self.embed_features = embed_features
        # film layer
        activation = nn.SiLU()
        dropout = kwargs.get("dropout", 0.0)
        width = kwargs.get("width", 64)
        depth = kwargs.get("depth", 3)
        self.film = MLP(
            in_dim=self.embed_features,
            out_dim=2 * self.target_channels,
            width=[width] * depth,
            activations=[activation] * depth + [None],
            dropout=dropout,
        )

    def _check_inputs(self, target: torch.Tensor, embed: torch.Tensor):
        """check the inputs"""
        assert (
            target.shape[1] == self.target_channels
        ), f"expected target of shape (batch_size, {self.target_channels}, *dims)"
        assert (
            embed.shape[1] == self.embed_features
        ), f"expected {self.embed_features} embedding features, got {embed.shape[1]}"
        assert (
            target.shape[2:] == embed.shape[2:]
        ), "expected the same dims for the embed and target"

    def forward(self, target: torch.Tensor, embed: torch.Tensor):
        """forward pass
        Args:
            target (torch.Tensor):
                target tensor of shape (batch_size, target_channels, *dims)
                to apply film via embed.
            embed (torch.Tensor):
                embed tensor of shape (batch_size, embed_features, *dims)
        """
        batch_size, _, *dims = target.shape
        self._check_inputs(target=target, embed=embed)
        # flatten the embedding
        embed_flat = embed.flatten(2).transpose(1, 2)
        # compute gamma and beta
        gamma, beta = self.film(embed_flat).chunk(2, dim=-1)
        gamma = gamma.transpose(1, 2).view(batch_size, self.target_channels, *dims)
        beta = beta.transpose(1, 2).view(batch_size, self.target_channels, *dims)
        # apply film
        target_mod = target * (1.0 + gamma) + beta
        assert target_mod.shape == target.shape, "Invalid modulation"
        return target_mod


class CrossAttention(nn.Module):
    """
    Memory-efficient multi-head cross-attention.

    Supports queries (e.g., W-domain features) attending to keys/values
    from another domain (e.g., A-domain features).
    TODO:
        Add multi-head
    """

    def __init__(
        self, dim_q: int, dim_kv: int, dim_out=None, num_heads: int = 1, **kwargs
    ):
        super(CrossAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.dim_out = dim_out or self.dim_q
        assert (
            self.dim_out % num_heads == 0
        ), "embed_dim_out must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = self.dim_out // self.num_heads
        self.scale = self.head_dim**-0.5

        self.query_proj = nn.Linear(self.dim_q, self.dim_out)
        self.key_proj = nn.Linear(self.dim_kv, self.dim_out)
        self.value_proj = nn.Linear(self.dim_kv, self.dim_out)

        self.out_proj = nn.Linear(self.dim_out, self.dim_out)
        self.dropout = nn.Dropout(kwargs.get("dropout", 0.0))

    def _check_input(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        """check the input"""
        pass
        assert query.shape[-1] == self.dim_q, "invalid query"
        assert key.shape[-1] == self.dim_kv, "invalid key"
        assert value.shape[-1] == self.dim_kv, "invalid value"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            query (torch.Tensor):
                features from target (this will attend)
            key (torch.Tensor):
                keys form the content to attend to
            value (torch.Tensor):
                modulating values
            attn_mask (torch.Tensor):
                optional attention mask

        Returns:
            out (torch.Tensor):
                Attented output
            attn_weights (torch.Tensor):
                Attention weights
        """
        batch_size = query.shape[0]
        Nq = query.shape[1]  # number of tokens in query
        Nk = key.shape[1]  # number of tokens in key
        Nv = value.shape[1]  # number of tokens in value
        # check the input
        self._check_input(query=query, key=key, value=value, attn_mask=attn_mask)
        # project
        Q = self.query_proj(query)  # (batch_size, Nq, dim_out)
        K = self.key_proj(key)  # (batch_size, Nk, dim_out)
        V = self.value_proj(value)  # (batch_size, Nk, dim_out)
        # Split heads: (B, num_heads, N, head_dim)
        Q = Q.view(batch_size, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, Nv, self.num_heads, self.head_dim).transpose(1, 2)
        # FlashAttention-enabled call (batch_size, num_heads, Nq, head_dim)
        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, dropout_p=0.0
        )
        # Collapse head dimension
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )
        out = self.out_proj(out)

        return out


class RBFEncoding(nn.Module):
    """
    Encoding using RBF kernels.
    Convert the domain of shape (batch_size, N, ndim) to (batch_size, N, n_centers)
    """

    def __init__(
        self,
        ndim: int,
        n_centers: int = 10,
        learnable_centers: bool = True,
        domain_min: float = 0.0,
        domain_max: float = 1.0,
    ):
        super(RBFEncoding, self).__init__()
        self.ndim = ndim
        self.n_centers = n_centers
        self.output_features = self.n_centers  # same as num centers
        self.domain_min = domain_min
        self.domain_max = domain_max
        # initialize the centers
        centers = self._initialize_centers()

        if learnable_centers:
            self.centers = nn.Parameter(centers)
            self.scale = nn.Parameter(torch.rand(self.n_centers))
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("scale", torch.rand(self.n_centers))

    def _initialize_centers(self):
        if self.ndim == 1:
            # uniform spacing for 1D
            centers = torch.linspace(
                self.domain_min, self.domain_max, self.n_centers
            ).unsqueeze(1)
        else:
            # For higher dimensions, use quasi-random sampling
            # Grid-based initialization for better coverage
            per_dim = int(math.ceil(self.n_centers ** (1.0 / self.ndim)))
            grids = [
                torch.linspace(self.domain_min, self.domain_max, per_dim)
                for _ in range(self.ndim)
            ]
            mesh = torch.meshgrid(*grids, indexing="ij")
            centers = torch.stack([g.flatten() for g in mesh], dim=1)[: self.n_centers]

            # Add small random perturbation to avoid perfect grid
            centers += (
                torch.randn_like(centers) * (self.domain_max - self.domain_min) * 0.05
            )
            centers = torch.clamp(centers, self.domain_min, self.domain_max)

        assert centers.shape == (self.n_centers, self.ndim)

        return centers

    def _check_input(self, coords: torch.Tensor):
        assert coords.ndim == 3, f"exptected (batch_size, N, ndim), got {coords.shape}"

    def _get_rbf_encoding(self, coords):
        batch_size, N = coords.shape[0], coords.shape[1]
        # compute diff (batch_size, N, num_center, ndim)
        diff = coords.unsqueeze(2) - self.centers.view(1, 1, self.n_centers, self.ndim)
        diff_nsq = (diff**2).sum(dim=-1)
        # compute rbf encoding
        scale = F.softplus(self.scale).view(1, 1, -1) + 1e-6
        coords_encoding = torch.exp(-0.5 * diff_nsq / scale)
        assert coords_encoding.shape == (batch_size, N, self.n_centers)
        return coords_encoding

    def forward(self, coords: torch.Tensor):
        """forward pass"""
        self._check_input(coords)
        return self._get_rbf_encoding(coords)


class FourierEncoding(nn.Module):
    """
    Encoding using fourier encoding
    Convert the domain of shape (batch_size, N, ndim) to (batch_size, N, n_centers)
    """

    def __init__(
        self, ndim: int, n_fourier_modes: int = 5, learnable_modes: bool = True
    ):
        super(FourierEncoding, self).__init__()
        self.ndim = ndim
        self.n_fourier_modes = n_fourier_modes
        self.output_features = (
            2 * self.n_fourier_modes
        )  # (sin(2 * pi * x), cos( 2 * pi * x))
        # initialize the modes
        modes = self._initialize_modes()
        if learnable_modes:
            self.modes = nn.Parameter(modes)
        else:
            self.register_buffer("modes", modes)

    def _initialize_modes(self):
        modes = torch.randn(self.ndim, self.n_fourier_modes)
        return modes

    def _check_input(self, coords: torch.Tensor):
        assert coords.ndim == 3, f"exptected (batch_size, N, ndim), got {coords.shape}"
        assert (
            coords.shape[-1] == self.ndim
        ), f"expected {self.ndim} features, got {coords.shape[-1]}"

    def _get_fourier_encoding(self, coords):
        batch_size = coords.shape[0]
        proj = 2.0 * math.pi * coords @ self.modes
        sin_enc = torch.sin(proj)
        cos_enc = torch.cos(proj)
        coords_encoding = torch.cat([sin_enc, cos_enc], dim=-1)
        assert coords_encoding.shape == (
            batch_size,
            coords.shape[1],
            self.output_features,
        )
        return coords_encoding

    def forward(self, coords: torch.Tensor):
        """forward pass"""
        self._check_input(coords)
        return self._get_fourier_encoding(coords)
