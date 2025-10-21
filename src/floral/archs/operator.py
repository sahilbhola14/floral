# src/floral/archs/operator.py
"""
operator modules
adapted from https://github.com/yzshi5/SPL_OFM/blob/main/models/fno.py
modified to account for fourier time embedding and conditional inputs.
"""
import torch
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv
from floral.utils import check_keys, printer
from .embedding import CrossAttention, MLP, build_pos_encoder, conv_nd


def get_vector_field_operator(operator_config: dict):
    """build the operator modules
    Args:
        operator_config (dict):
            configuration for building the operator
    """
    # check the required keys
    required_keys = ["field", "condition"]
    check_keys(operator_config, required_keys)
    # create model
    return VectorField(
        field_config=operator_config["field"],
        condition_config=operator_config["condition"],
    )


class SpectralBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple,
        cond_dim: int | None = None,
        cond_method: str = "attention",
    ):
        super(SpectralBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.ndim = len(self.n_modes)
        self.cond_dim = cond_dim
        self.apply_condition = self.cond_dim is not None

        # spectral layer
        self.spectral_layer = SpectralConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_modes=self.n_modes,
        )
        # layer norm
        self.norm = nn.LayerNorm(self.out_channels)
        # condition layer
        if self.apply_condition:
            self.condition_layer = self._build_condition_layer(
                cond_dim=cond_dim, cond_method=cond_method
            )

    def _build_condition_layer(
        self, cond_dim: int = None, cond_method: str = "cross_attn"
    ):
        """build the conditon layer"""
        if cond_dim == "cross_attn":
            condition_layer = CrossAttention(
                dim_q=self.out_channels, dim_kv=self.cond_dim
            )
        else:
            raise ValueError(f"condition method: {cond_method} not implemented")

        return condition_layer

    def forward(self, x, cond=None):
        x = self.spectral_layer(x)
        x = self.norm(x)
        if self.apply_condition:
            x = self.condition_layer(x, cond, cond)
        return x


class FNOBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple,
        cond_dim: int | None = None,
        activation: str = "gelu",
        skip: str = "linear",
    ):
        cond_dim = self.condition_channels
        super(FNOBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.cond_dim = cond_dim
        self.cond_
        self.ndim = len(self.n_modes)
        # spectral layers
        self.spectral_layer = self._build_spectral_module()
        # skip layers
        self.skip = self._build_skip_module()
        # activation
        self.activation = self._get_activation(activation)

    def _build_spectral_module(self):
        return SpectralBlock(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_modes=self.n_modes,
        )

    def _build_skip_module(self):
        skip = conv_nd(
            self.ndim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
        )
        return skip

    def _get_activation(self, activation: str):
        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            printer(f"activation: f{activation} not implemented, applying identity")
            act = nn.Identity()
        return act

    def forward(self, x, cond=None):
        """apply forward"""
        x_spec = self.spectral_layer(x)  # (B, out_channels, *dims)
        if self.skip is not None:
            x_skip = self.skip(x)
            x = x_spec + x_skip
        x = self.activation(x)
        return x


class FNO(nn.Module):
    """FNO architecture with latent cross-attention
    Args:
      in_channels: Number of input channels
      hidden_channels: Number of hidden channels
      out_channels: Number of output channels
      n_modes: Number of Fourier modes (tuple for each dimension)
      n_layers: Number of FNO layers
      cond_dim: Dimension of conditioning vector (None to disable conditioning)
      activation: Activation function ('gelu', 'relu', 'tanh')
      skip: Skip connection type ('linear', 'identity', None)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_modes: tuple,
        n_layers: int = 4,
        cond_dim: int = None,
        activation: str = "gelu",
        skip: str = "linear",
    ):
        super(FNO, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.cond_dim = cond_dim
        self.activation = activation
        self.skip = skip

        assert isinstance(self.n_modes, tuple), "expected n_modes to be a tuple"
        self.ndim = len(self.n_modes)

        # lifting layer
        self.lifting = self._build_lifting_module()
        # spectral layers
        self.spectral_layers = self._build_spectral_module()
        # projection layer
        self.projection = self._build_projection_module()

    def _build_lifting_module(self):
        """build the lifting module"""
        # lifting layer
        lifting = conv_nd(
            self.ndim,
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
        )
        return lifting

    def _build_spectral_module(self):
        """build the spectral module"""
        spectral_layers = nn.ModuleList(
            [
                FNOBlock(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    n_modes=self.n_modes,
                    cond_dim=self.cond_dim,
                    activation=self.activation,
                    skip=self.skip,
                )
                for _ in range(self.n_layers)
            ]
        )
        return spectral_layers

    def _build_projection_module(self):
        """projection module"""
        projection = nn.Sequential(
            conv_nd(
                self.ndim,
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=1,
            ),
            nn.GELU(),
            conv_nd(
                self.ndim,
                in_channels=self.hidden_channels,
                out_channels=self.out_channels,
                kernel_size=1,
            ),
        )
        return projection

    def forward(self, x, cond=None):
        """forward pass"""
        # lifting
        x = self.lifting(x)
        # apply spectral layers
        for layer in self.spectral_layers:
            x = layer(x, cond)
        # project back
        x = self.projection(x)
        return x


class SpatialDecoder(nn.Module):
    """Decode latent features to spatial field using coordinate-based queries"""

    def __init__(self, latent_channels: int, field_ndim: int, **kwargs: dict):
        super(SpatialDecoder, self).__init__()
        self.field_ndim = field_ndim
        # coordinate encoder
        embed_dim = kwargs.get("embed_dim", 128)
        self.coord_encoder = MLP(
            in_dim=self.field_ndim,
            width=[64, 64],
            out_dim=embed_dim,
            activations=[nn.ReLU(), nn.ReLU(), None],
            dropout=kwargs.get("dropout", 0.0),
        )
        # attention
        self.cross_attention = CrossAttention(
            dim_q=embed_dim,
            dim_kv=latent_channels,
            num_heads=kwargs.get("num_heads", 8),
        )
        # final projection
        self.out_projection = MLP(
            in_dim=embed_dim,
            width=[latent_channels],
            out_dim=latent_channels,
            activations=[nn.GELU(), None],
            dropout=kwargs.get("dropout", 0.0),
        )

    def forward(self, latent_features: torch.Tensor, field_domain: torch.Tensor):
        """forward pass"""
        raise NotImplementedError
        # flatten the field domain
        field_domain_flat = field_domain.flatten(2).transpose(1, 2)
        # encode the field coordinates (queries) (B, Nq, embed_dim)
        coord_queries = self.coord_encoder(field_domain_flat)
        print(latent_features[0])
        # cross attention (B, Nq, embed_dim)
        attended_coords = self.cross_attention(
            query=coord_queries, key=latent_features, value=latent_features
        )
        # output
        out = self.out_projection(
            attended_coords
        )  # (batch_size, N_field, latent_channels)
        # reshape output
        out = out.transpose(1, 2)

        return out


class VectorField(nn.Module):
    def __init__(
        self,
        field_config: dict,
        condition_config: dict,
        t_scaling: float = 1.0,
        **kwargs,
    ):
        super(VectorField, self).__init__()
        # intial check
        required_keys = ["channels", "ndim"]
        check_keys(field_config, required_keys)
        # set attributes
        self.t_scaling = t_scaling
        self.field_channels = field_config.get("channels")
        self.field_hidden_channels = field_config.get("hidden_channels", 128)
        self.field_ndim = field_config.get("ndim")
        self.field_modes = field_config.get("modes", 32)
        self.condition_channels = condition_config.get("channels")
        self.condition_out_channels = condition_config.get("out_channels")
        # domain encoder
        (
            self.field_domain_encoder,
            self.field_domain_embed_dim,
        ) = self._build_pos_encoder(pos_encoder=kwargs.get("pos_encoder", "fourier"))
        # field processor
        self.field_model = self._build_field_module()

    def _build_field_module(self):
        """build the field processor operator"""
        # same modes in each dimension
        n_modes = (self.field_modes,) * self.field_ndim
        # in_channels for the operator
        in_channels = (
            self.field_channels
            + self.field_domain_embed_dim
            + self.condition_channels
            + 1
        )
        return FNO(
            in_channels=in_channels,
            hidden_channels=self.field_hidden_channels,
            out_channels=self.field_channels,
            n_modes=n_modes,
            cond_dim=self.condition_channels,
        )

    def _build_pos_encoder(self, pos_encoder: str = "rbf"):
        """build the position encoder"""
        field_domain_encoder = build_pos_encoder(
            pos_encoder=pos_encoder,
            ndim=self.field_ndim,
        )
        domain_embed_dim = field_domain_encoder.output_features
        return field_domain_encoder, domain_embed_dim

    def forward(
        self,
        psi: torch.Tensor,
        condition: torch.Tensor,
        field_domain: torch.Tensor,
        condition_domain: torch.Tensor,
        t: torch.Tensor,
    ):
        """forward pass
        Args:
            psi (torch.Tensor):
                samples from the condition path of shape
                (batch_size, field_channels, *field_dims)
            condition (torch.Tensor):
                conditions of shape (batch_size, condition_channels, *condition_dims)
            field_domain (torch.Tensor):
                domain of shape (1, field_domain_channels, *field_domain_dims)
            condition_domain (torch.Tensor):
                domain of shape (1, condition_domain_channels, *condition_domain_dims)
            t (torch.Tensor):
                samples of time of shape (batch_size, 1)
        """
        batch_size, _, *field_dims = psi.shape
        # time embedding
        t_scaled = t / self.t_scaling
        t_embed = t_scaled.view(batch_size, 1, *([1] * self.field_ndim)).expand(
            -1, -1, *field_dims
        )
        # domain embedding
        field_domain_expd = field_domain.expand(batch_size, *field_domain.shape[1:])
        field_domain_flat = field_domain_expd.flatten(2).transpose(1, 2)
        field_domain_embed = self.field_domain_encoder(coords=field_domain_flat)
        field_domain_embed = field_domain_embed.transpose(1, 2).view(
            batch_size, -1, *field_dims
        )
        # input
        inp_vt = torch.cat((psi, field_domain_embed, condition, t_embed), dim=1)
        vt = self.field_model(inp_vt)
        return vt
