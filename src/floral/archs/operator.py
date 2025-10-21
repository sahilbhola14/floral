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
from .embedding import CrossAttention, build_pos_encoder, conv_nd


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
        num_latents: int = 128,
        latent_dim: int = 256,
        **kwargs: dict,
    ):
        super(SpectralBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.ndim = len(self.n_modes)
        self.cond_dim = cond_dim
        self.apply_condition = self.cond_dim is not None
        self.num_latents = num_latents
        self.latent_dim = latent_dim

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
            # latent queries
            self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim))
            # cross attention: latent attends to the condition
            self.cond_cross_attn = CrossAttention(
                dim_q=self.latent_dim,
                dim_kv=self.cond_dim,
                num_heads=kwargs.get("num_heads", 8),
            )
            # cross attention: field attends to the latents
            self.latent_cross_attn = CrossAttention(
                dim_q=self.out_channels,
                dim_kv=self.latent_dim,
                num_heads=kwargs.get("num_heads", 8),
            )

    def _condition_forward(self, x, cond):
        """apply condition to the output via latent cross attention"""
        assert (
            cond is not None
        ), "need to provide condition for doing condition forward pass"
        batch_size, _, *dims = x.shape
        # expand latents (batch_size, num_latents, latent_dim)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        # flatten condition for attention (batch_size, Nc, cond_dim)
        cond_flat = cond.flatten(2).transpose(1, 2)
        # latents attend to the condition (batch_size, num_latents, latent_dim)
        latents = self.cond_cross_attn(query=latents, key=cond_flat, value=cond_flat)
        # flatten field for attention (batch_size, Nx, out_channels)
        x_flat = x.flatten(2).transpose(1, 2)
        # field attends to the latents
        x_attn = self.latent_cross_attn(query=x_flat, key=latents, value=latents)
        # reshape to original
        x_attn = x_attn.transpose(1, 2).view(batch_size, -1, *dims)

        return x + x_attn

    def forward(self, x, cond=None):
        # apply the spectral layer
        x = self.spectral_layer(x)
        # normalize
        x = self.norm(x)
        if self.apply_condition:
            x = self._condition_forward(x=x, cond=cond)
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
        super(FNOBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.cond_dim = cond_dim
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
            cond_dim=self.cond_dim,
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
        x_spec = self.spectral_layer(x=x, cond=cond)  # (B, out_channels, *dims)
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
            x = layer(x=x, cond=cond)
        # project back
        x = self.projection(x)
        return x


class VectorField(nn.Module):
    def __init__(
        self,
        field_config: dict,
        condition_config: dict,
        **kwargs,
    ):
        super(VectorField, self).__init__()
        # intial check
        required_keys = ["channels", "ndim"]
        check_keys(field_config, required_keys)
        check_keys(condition_config, required_keys)
        # set attributes
        self.field_channels = field_config.get("channels")
        self.field_ndim = field_config.get("ndim")
        self.field_hidden_channels = field_config.get("hidden_channels", 128)
        self.field_modes = field_config.get("modes", 32)
        self.condition_channels = condition_config.get("channels")
        self.condition_ndim = condition_config.get("ndim")

        # domain encoders
        (self.field_domain_encoder, self.field_domain_embed_dim,) = build_pos_encoder(
            pos_encoder=kwargs.get("pos_encoder", "fourier"),
            ndim=self.field_ndim,
        )

        (
            self.condition_domain_encoder,
            self.condition_domain_embed_dim,
        ) = build_pos_encoder(
            pos_encoder=kwargs.get("pos_encoder", "fourier"),
            ndim=self.condition_ndim,
        )
        # time encoder
        self.time_encoder, self.time_embed_dim = build_pos_encoder(
            pos_encoder=kwargs.get("time_encoder", "fourier"),
            ndim=1,
            learnable_modes=False,
        )
        # operator
        self.field = self._build_field_module()

    def _build_field_module(self):
        """build the field processor operator
        Vector field operator with inputs
        1. psi:
            samples from the conditional flow
        2. field_domain:
            domain for the field at which we need to generate the predition
        3. condition:
            additional conditions
        4. time:
            time samples
        """
        # same modes in each dimension
        n_modes = (self.field_modes,) * self.field_ndim
        # in_channels to the field
        in_channels = (
            self.field_channels + self.field_domain_embed_dim + self.time_embed_dim
        )
        # condition dims
        cond_dim = self.condition_channels + self.condition_domain_embed_dim
        # field model
        field_model = FNO(
            in_channels=in_channels,
            hidden_channels=self.field_hidden_channels,
            out_channels=self.field_channels,
            n_modes=n_modes,
            cond_dim=cond_dim,
        )

        return field_model

    def _check_inputs(
        self,
        field: torch.Tensor,
        condition: torch.Tensor,
        field_domain: torch.Tensor,
        condition_domain: torch.Tensor,
        t: torch.Tensor,
    ):
        """check inputs"""
        assert (
            field.ndim == self.field_ndim + 2
        ), f"expected {self.field_ndim + 2}-D field, got {field.ndim}"
        assert (
            field.shape[1] == self.field_channels
        ), f"expected {self.field_channels} field channels, got {field.shape[1]}"
        assert (
            condition.ndim == self.condition_ndim + 2
        ), f"expected {self.condition_ndim + 2}-D condition, got {condition.ndim}"
        assert (
            condition.shape[1] == self.condition_channels
        ), f"expected {self.condition_channels} condition channels, "
        f" got {condition.shape[1]}"
        assert (
            t.ndim == 2 and t.shape[1] == 1
        ), f"expected time of shape (batch_size, 1), got {t.shape}"
        assert field_domain.ndim == 3, "expected field domain of shape "
        f"(batch_size, {self.field_ndim}, {field.shape[2:]})"
        assert condition_domain.ndim == 3, "expected condition domain of shape "
        f"(batch_size, {self.condition_ndim}, {condition.shape[2:]}"

    def _get_time_embedding(self, t: torch.Tensor, field_dims: list):
        """time embedding
        Args:
            t (torch.Tensor):
        """
        batch_size = t.shape[0]
        t_embed = self.time_encoder(t.unsqueeze(1)).squeeze(1)
        t_embed = t_embed.view(batch_size, -1, *([1] * self.field_ndim)).expand(
            -1, -1, *field_dims
        )
        return t_embed

    def _get_field_domain_embedding(self, field_domain: torch.Tensor):
        """field domain embedding"""
        batch_size = field_domain.shape[0]
        field_domain_flat = field_domain.flatten(2).transpose(1, 2)
        domain_embed = (
            self.field_domain_encoder(field_domain_flat)
            .transpose(1, 2)
            .view(batch_size, -1, *field_domain.shape[2:])
        )
        return domain_embed

    def _get_condition_domain_embedding(self, condition_domain: torch.Tensor):
        """condition domain embedding"""
        batch_size = condition_domain.shape[0]
        condition_domain_flat = condition_domain.flatten(2).transpose(1, 2)
        domain_embed = (
            self.condition_domain_encoder(condition_domain_flat)
            .transpose(1, 2)
            .view(batch_size, -1, *condition_domain.shape[2:])
        )
        return domain_embed

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
        _, _, *condition_dims = condition.shape
        # check inputs
        self._check_inputs(
            field=psi,
            condition=condition,
            field_domain=field_domain,
            condition_domain=condition_domain,
            t=t,
        )
        # time embedding (batch_size, embed_dim, *field_dims)
        t_embed = self._get_time_embedding(t=t, field_dims=field_dims)
        # field domain embedding (batch_size, embed_dim, *field_dims)
        field_domain_embed = self._get_field_domain_embedding(field_domain).expand(
            batch_size, *([-1] * (field_domain.ndim - 1))
        )
        # condition domain embedding (batch_size, embed_dim, *condiiton_dims)
        condition_domain_embed = self._get_condition_domain_embedding(
            condition_domain
        ).expand(batch_size, *([-1] * (condition_domain.ndim - 1)))
        # input field
        inp_vt = torch.cat((psi, field_domain_embed, t_embed), dim=1)
        # condition
        cond_vt = torch.cat((condition, condition_domain_embed), dim=1)
        vt = self.field(x=inp_vt, cond=cond_vt)
        return vt
