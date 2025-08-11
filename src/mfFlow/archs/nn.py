import math
import torch
import torch.nn as nn
from .encoding import (
    FiLM,
    MLP,
    RBFFiLM,
    conv_nd,
    SpatialAttentionPooling,
    SpatialAdaptivePooling,
)


def zero_module(module: nn.Module):
    """zero out the parameters of a module and return it"""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization1D(in_features):
    """Layer Normalize"""
    return LayerNorm32(in_features)


def normalization2D(channels):
    """Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def get_embedding_modules(
    nx: int,
    nc: int,
    nd: int,
    latent_dim: int,
    time_embed_freq: int,
    num_centers: int,
    field_data: bool = False,
    **kwargs,
):
    """Wrapper to get the modules for StateEmbedding, ConditionEmbedding,
    FusionEmbedding, and DomainEmbedding"""
    # state embedding
    state_embedding = StateEmbedding(
        nx=nx,
        latent_dim=latent_dim,
        time_embed_freq=time_embed_freq,
        dropout=kwargs.get("dropout", 0.0),
        **kwargs,
    )
    # condition embedding
    if field_data:
        condition_embedding = Condition2DEmbedding(
            nc=nc,
            latent_dim=latent_dim,
            time_embed_freq=time_embed_freq,
            dropout=kwargs.get("dropout", 0.0),
            **kwargs,
        )
    else:
        condition_embedding = Condition1DEmbedding(
            nc=nc,
            latent_dim=latent_dim,
            time_embed_freq=time_embed_freq,
            dropout=kwargs.get("dropout", 0.0),
            **kwargs,
        )
    # fusion embedding
    fusion_embedding = FusionEmbedding(
        latent_dim=latent_dim,
        time_embed_freq=time_embed_freq,
        dropout=kwargs.get("dropout", 0.0),
        **kwargs,
    )

    # domain embedding
    domain_embedding = RBFFiLM(
        num_centers=num_centers,
        latent_dim=latent_dim,
        nd=nd,
        nx=nx,
        dropout=kwargs.get("dropout", 0.0),
        **kwargs,
    )

    embedding = {
        "state_embedding": state_embedding,
        "condition_embedding": condition_embedding,
        "fusion_embedding": fusion_embedding,
        "domain_embedding": domain_embedding,
    }
    return embedding


class LayerNorm32(nn.LayerNorm):
    """Layer normalization for 1D inputs with arbitrary precision"""

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class Res2DBlock(nn.Module):
    """Residual block for 2D inputs"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super(Res2DBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout

        #  input layer
        self.input_layer = (
            nn.Sequential(
                normalization2D(self.in_channels),
                conv_nd(
                    dims=2,
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    padding=1,
                ),
            )
            if self.in_channels != self.out_channels
            else nn.Identity()
        )

        # embedding
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    normalization2D(self.out_channels),
                    conv_nd(
                        dims=2,
                        in_channels=self.out_channels,
                        out_channels=self.out_channels,
                        kernel_size=3,
                        padding=1,
                    ),
                )
                for _ in range(2)
            ]
        )
        # film
        self.film_layers = nn.ModuleList(
            [FiLM(self.time_embed_dim, self.out_channels) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """forward pass"""
        # input layer
        skip = self.input_layer(x)
        out = skip
        # apply transformations
        for conv_layer, film_layer in zip(self.conv_layers, self.film_layers):
            out = conv_layer(out)
            out = film_layer(out, t)
        return out + skip


class SkipConnection(nn.Module):
    """Skip connection with regularization"""

    def __init__(self, latent_dim, hidden_dim=128, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.layers = MLP(
            in_dim=self.latent_dim,
            out_dim=self.latent_dim,
            width=[hidden_dim, hidden_dim],
            activations=[nn.SiLU(), nn.SiLU(), None],
            norm="layer",
            dropout=self.dropout,
        )

        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return self.layers(x) * self.residual_weight + x


class Condition1DEmbedding(nn.Module):
    """Class for 1D condition embedding

    This class embeds an input of (batch_size, nc) into (batch_size, latent_dim)
    Attibutes:
        nc (int): Dimensionality of the conditional input
        latent_dim (int): Dimensionality of the encodedlatent space
        time_emb_freq (int): Frequency of the time embedding
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        nc: int,
        latent_dim: int,
        time_embed_freq: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super(Condition1DEmbedding, self).__init__()
        self.nc = nc
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.time_embed_dim = 2 * time_embed_freq  # (sin(), cos())
        # skip connection
        if self.nc != self.latent_dim:
            self.skip = MLP(
                in_dim=self.nc,
                width=[],
                activations=[None],
                out_dim=self.latent_dim,
                dropout=self.dropout,
            )
        else:
            self.skip = nn.Identity()

        # condition embedding
        self.condition_embedding = MLP(
            in_dim=self.nc,
            width=[32, 32],
            out_dim=self.latent_dim,
            activations=[nn.SiLU(), nn.SiLU(), None],
            norm="layer",
            dropout=self.dropout,
        )

        # FiLM embedding
        self.film = FiLM(self.time_embed_dim, self.latent_dim)

    def forward(self, condition: torch.Tensor, time_embed: torch.Tensor):
        """forward pass of the conditional embedding

        Args:
            condition (torch.Tensor): Input tensor of shape (batch_size, nc)
            time_embed (torch.Tensor): Time embedding tensor of shape
            (batch_size, time_embed_freq * 2)

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, latent_dim)
        """
        # apply skip
        skip = self.skip(condition)
        # apply conditon embedding
        condition_embedding = self.condition_embedding(condition) + skip
        # apply FiLM
        out = self.film(
            condition_embedding.unsqueeze(-1).unsqueeze(-1), time_embed
        ).squeeze()
        return out


class Condition2DEmbedding(nn.Module):
    """Class for 2D condition embedding

    This class embeds an input of (batch_size, nc) into (batch_size, latent_dim)
    Attibutes:
        nc (int): Dimensionality of the conditional input
        latent_dim (int): Dimensionality of the encodedlatent space
        time_emb_freq (int): Frequency of the time embedding
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        nc: int,
        latent_dim: int,
        time_embed_freq: int,
        channel_mult=None,
        base_channels: int = 32,  # multiplier for the number of channels
        in_channels: int = 1,  # number of input channels in the condition
        feature_pool_type: str = "adaptive",  # options: adaptive, attention
        pool_type: str = "max",  # options: max-> MaxPool2D, avg -> AvgPool2d
        num_attention_heads: int = 1,
        attention_embed_dim: int = 64,
        dropout: float = 0.1,
        **kwargs,
    ):
        super(Condition2DEmbedding, self).__init__()
        self.nc = nc
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.time_embed_dim = 2 * time_embed_freq  # (sin(), cos())
        self.in_channels = in_channels
        self.feature_pool_type = feature_pool_type
        self.pool_type = pool_type
        self.num_attention_heads = num_attention_heads
        self.field_shape = (int(math.sqrt(self.nc)), int(math.sqrt(self.nc)))
        assert (
            self.field_shape[0] * self.field_shape[1] == self.nc
        ), "Only square fields currently supported"

        if channel_mult is None:
            image_size = self.field_shape[0]
            if image_size == 64:
                self.channel_mult = (1, 2, 3, 4)
            elif image_size == 32:
                self.channel_mult = (1, 2, 2, 2)
            elif image_size == 50:
                self.channel_mult = (1, 2, 4)
            else:
                raise ValueError(
                    "Unsupported size: {}. Supported sizes are 32, 50, 64.".format(
                        image_size
                    )
                )
        else:
            self.channel_mult = channel_mult

        # input block to process the condition
        ch = input_ch = int(self.channel_mult[0] * base_channels)
        self.input_blocks = conv_nd(
            dims=2, in_channels=in_channels, out_channels=ch, kernel_size=3, padding=1
        )

        # encoding blocks
        self.residual_blocks = nn.ModuleList()
        self.feature_blocks = nn.ModuleList()
        self.pool_blocks = nn.ModuleList()
        for mult in self.channel_mult:
            output_ch = mult * base_channels
            # residual blocks
            self.residual_blocks.append(
                Res2DBlock(
                    in_channels=input_ch,
                    out_channels=output_ch,
                    time_embed_dim=self.time_embed_dim,
                    dropout=self.dropout,
                )
            )
            # feature blocks
            if self.feature_pool_type == "adaptive":
                self.feature_blocks.append(
                    SpatialAdaptivePooling(
                        in_channels=output_ch,
                        latent_dim=self.latent_dim,
                        dropout=self.dropout,
                    )
                )
            elif self.feature_pool_type == "attention":
                self.feature_blocks.append(
                    SpatialAttentionPooling(
                        in_channels=output_ch,
                        latent_dim=self.latent_dim,
                        num_attention_heads=self.num_attention_heads,
                        embed_dim=attention_embed_dim,
                        dropout=self.dropout,
                    )
                )
            else:
                raise ValueError(f"Invalid feature pool type {self.feature_pool_type}")

            # pool blocks
            if self.pool_type == "max":
                self.pool_blocks.append(nn.MaxPool2d(2))
            elif self.pool_type == "avg":
                self.pool_blocks.append(nn.AdaptiveAvgPool2d(1))
            else:
                raise ValueError(f"Invalid pool type {self.pool_type}")

            # update the input channels
            input_ch = output_ch

        # spatial feature weights
        self.spatial_feature_weights = nn.Parameter(
            torch.ones(len(self.channel_mult)) / len(self.channel_mult)
        )

        # Time FiLM
        self.film = FiLM(self.time_embed_dim, self.latent_dim)

        # output projection
        self.out_proj = nn.Sequential(
            normalization1D(self.latent_dim),
            MLP(
                in_dim=self.latent_dim,
                out_dim=self.latent_dim,
                width=[32, 32],
                activations=[nn.SiLU(), nn.SiLU(), None],
                norm="layer",
                dropout=self.dropout,
            ),
        )

    def forward(self, condition: torch.Tensor, time_embed: torch.Tensor):
        """forward pass"""
        # reshape the condition to field (B, C, H, W)
        condition = condition.view(-1, self.in_channels, *self.field_shape)
        # process through the input block (B, base_channels, H, W)
        condition = self.input_blocks(condition)
        # process throught the residual and pool blocks
        out = condition
        spatial_features = []
        for ii, (residual_block, feature_block, pool_block) in enumerate(
            zip(self.residual_blocks, self.feature_blocks, self.pool_blocks)
        ):
            # residual block
            out = residual_block(out, time_embed)
            # store spatial features
            spatial_features.append(feature_block(out))
            # pooling
            if ii < len(self.channel_mult) - 1:
                out = pool_block(out)

        # weighted spatial features
        weighted_features = [
            self.spatial_feature_weights[i] * feat
            for i, feat in enumerate(spatial_features)
        ]
        fused_features = torch.stack(weighted_features).sum(dim=0)

        # FiLM
        out = self.film(
            fused_features.unsqueeze(-1).unsqueeze(-1), time_embed
        ).squeeze()

        return self.out_proj(out)


class StateEmbedding(nn.Module):
    """Class for embedding state information
    This class embeds an input of (batch_size, nc) into (batch_size, latent_dim)

    Attibutes:
        nx (int): Dimensionality of the state input
        latent_dim (int): Dimensionality of the encodedlatent space
        time_emb_freq (int): Frequency of the time embedding
        hidden_dims (list): List of hidden layer sizes
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        nx: int,
        latent_dim: int,
        time_embed_freq: int,
        hidden_dims: list = [64, 128],
        dropout: float = 0.1,
        **kwargs,
    ):
        super(StateEmbedding, self).__init__()
        self.nx = nx
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.time_embed_dim = 2 * time_embed_freq  # (sin(), cos())
        self.hidden_dims = hidden_dims
        assert len(self.hidden_dims) == 2, "currently only 2 layer support."

        # skip connection
        if self.nx != self.hidden_dims[-1]:
            self.skip = nn.Linear(self.nx, self.hidden_dims[-1])

        # state embedding
        self.state_embedding = MLP(
            in_dim=self.nx,
            out_dim=self.hidden_dims[1],
            width=[self.hidden_dims[0]],
            activations=[nn.SiLU(), None],
            dropout=self.dropout,
            norm="layer",
        )

        # time projection
        self.time_proj = MLP(
            in_dim=self.time_embed_dim,
            out_dim=self.hidden_dims[1],
            width=[],
            activations=[nn.Tanh()],
        )

        # FiLM layers
        self.film_layers = nn.ModuleList(
            [
                FiLM(self.time_embed_dim, self.hidden_dims[1]),
                FiLM(self.time_embed_dim, self.hidden_dims[1]),
            ]
        )

        # Main processing layer
        self.main_layers = nn.ModuleList(
            [
                MLP(
                    in_dim=self.hidden_dims[1],
                    out_dim=self.hidden_dims[1],
                    width=[],
                    activations=[nn.SiLU()],
                    norm="layer",
                    dropout=self.dropout,
                )
                for _ in range(2)
            ]
        )

        # output projection
        self.out_proj = MLP(
            in_dim=self.hidden_dims[1],
            out_dim=self.latent_dim,
            width=[],
            activations=[None],
            dropout=self.dropout,
        )

    def forward(self, state: torch.Tensor, time_embed: torch.Tensor):
        """forward pass of the state embedding

        Args:
            state (torch.Tensor): Input tensor of shape (batch_size, nx)
            time_emb (torch.Tensor): Time embedding tensor of shape
            (batch_size, time_embed_freq * 2)

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, latent_dim)
        """
        # skip
        skip = self.skip(state)
        # state embedding
        state_embedding = self.state_embedding(state) + skip
        # time modulation
        out = state_embedding + self.time_proj(time_embed)
        # Apply FiLM modulation and residual layers
        for film, layer in zip(self.film_layers, self.main_layers):
            residual = out
            out = film(out.unsqueeze(-1).unsqueeze(-1), time_embed).squeeze()
            out = layer(out)
            out += residual
        return self.out_proj(out)


class FusionEmbedding(nn.Module):
    """Fuse the state and condition"""

    def __init__(
        self, latent_dim: int, time_embed_freq: int, dropout: float = 0.1, **kwargs
    ):
        super(FusionEmbedding, self).__init__()
        self.latent_dim = latent_dim
        self.dropout = kwargs.get("dropout", 0.1)
        self.time_embed_dim = 2 * time_embed_freq  # (sin(), cos())
        self.dropout = dropout

        # fusion layer
        self.fusion = MLP(
            in_dim=self.latent_dim,
            out_dim=self.latent_dim,
            width=[self.latent_dim],
            activations=[nn.SiLU(), None],
            norm="layer",
            dropout=self.dropout,
        )

        # skip connection
        self.skip = SkipConnection(latent_dim=self.latent_dim, dropout=self.dropout)

    def forward(
        self,
        state_embed: torch.Tensor,
        condition_embed: torch.Tensor,
        time_embed: torch.Tensor,
    ):
        """forward pass"""
        return self.skip(self.fusion(state_embed + condition_embed))
