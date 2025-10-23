# src/floral/archs/operator.py
"""
operator modules
adapted from https://github.com/yzshi5/SPL_OFM/blob/main/models/fno.py
modified to account for fourier time embedding,
conditional inputs, and domain embedding.
"""
import torch
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv
from floral.utils import check_keys, printer
from .embedding import ChannelFiLM, MLP, build_domain_encoder, conv_nd


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
        apply_condition: bool = False,
        **kwargs: dict,
    ):
        super(SpectralBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.ndim = len(self.n_modes)
        self.apply_condition = apply_condition

        # spectral layer
        self.spectral_layer = SpectralConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_modes=self.n_modes,
        )
        # layer norm
        self.norm = nn.LayerNorm(self.out_channels)

        # field domain embedding
        self.field_domain_encoder, field_domain_embed_dim = build_domain_encoder(
            ndim=self.ndim
        )
        self.field_domain_proj = MLP(
            in_dim=field_domain_embed_dim,
            out_dim=self.in_channels,
            activations=[nn.GELU()] * 2 + [None],
            width=[32, 32],
        )

        self.field_alpha_pre = nn.Parameter(torch.tensor(0.0))  # pre-gate for domain
        self.field_alpha_post = nn.Parameter(torch.tensor(0.0))  # post-gate for domain

        if self.apply_condition:
            self.cond_film = ChannelFiLM(
                target_channels=self.out_channels,
                embed_features=self.out_channels,
            )

    def forward(self, x, x_domain=None, cond=None):
        batch_size, _, *field_dims = x.shape
        # apply positional encoding
        if x_domain is not None:
            x_pos = self.field_domain_encoder(x_domain.flatten(2).transpose(1, 2))
            x_pos = (
                self.field_domain_proj(x_pos)
                .transpose(1, 2)
                .view(batch_size, -1, *field_dims)
            )
            x = x + self.field_alpha_pre * x_pos
        # apply the spectral layer
        x = self.spectral_layer(x)
        # re-inject positional info
        x = x + self.field_alpha_post * x_pos
        # apply film
        if self.apply_condition:
            assert x.shape == cond.shape, "assumes same domain for condition and field"
            x = self.cond_film(target=x, embed=cond)
        # normalize
        x = self.norm(x)
        return x


class FNOBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: tuple,
        activation: str = "gelu",
        apply_condition: bool = False,
    ):
        super(FNOBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.ndim = len(self.n_modes)
        self.apply_condition = apply_condition
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
            apply_condition=self.apply_condition,
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

    def forward(self, x, x_domain=None, cond=None):
        """apply forward"""
        x_spec = self.spectral_layer(
            x=x, x_domain=x_domain, cond=cond
        )  # (B, out_channels, *dims)
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
      cond_ndim: Dimensionality of the condition (1D/2D/3D)
      cond_channels: Number of input channels in the condition (None for no cond.)
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
        cond_ndim: int | None = None,
        cond_channels: int | None = None,
        activation: str = "gelu",
    ):
        super(FNO, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.cond_ndim = cond_ndim
        self.cond_channels = cond_channels
        self.activation = activation

        self.apply_condition = self.cond_channels is not None

        assert isinstance(self.n_modes, tuple), "expected n_modes to be a tuple"
        self.ndim = len(self.n_modes)

        # lifting layer(s)
        self.lifting = self._build_field_lifting_module()
        # spectral layers
        self.spectral_layers = self._build_spectral_module()
        # projection layer
        self.projection = self._build_projection_module()

        # condition
        if self.apply_condition:
            # lifting
            self.cond_lifting = self._build_cond_lifting_module()
            # film layers
            self.cond_film = ChannelFiLM(
                target_channels=self.hidden_channels,
                embed_features=self.hidden_channels,
            )

    def _build_field_lifting_module(self):
        """build the lifting module"""
        # lifting layer
        lifting = conv_nd(
            self.ndim,
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=1,
        )
        return lifting

    def _build_cond_lifting_module(self):
        """build the lifting module"""
        # lifting layer
        lifting = conv_nd(
            self.cond_ndim,
            in_channels=self.cond_channels,
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
                    activation=self.activation,
                    apply_condition=self.apply_condition,
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

    def forward(self, x, x_domain=None, cond=None):
        """forward pass"""
        # lifting
        x = self.lifting(x)
        # apply condition
        if self.apply_condition:
            # lifting
            cond = self.cond_lifting(cond)
            # FiLM
            x = self.cond_film(target=x, embed=cond)
        # apply spectral layers
        for layer in self.spectral_layers:
            x = layer(x=x, x_domain=x_domain, cond=cond)
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
        # time encoder
        self.time_encoder, self.time_embed_dim = build_domain_encoder(
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
            self.field_channels + self.time_embed_dim + self.condition_channels
        )
        # condition channels
        cond_channels = self.condition_channels

        # field model
        field_model = FNO(
            in_channels=in_channels,
            hidden_channels=self.field_hidden_channels,
            out_channels=self.field_channels,
            n_modes=n_modes,
            cond_ndim=self.condition_ndim,
            cond_channels=cond_channels,
        )

        return field_model

    def _check_inputs(
        self,
        field: torch.Tensor,
        condition: torch.Tensor,
        field_domain: torch.Tensor,
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

    def forward(
        self,
        psi: torch.Tensor,
        condition: torch.Tensor,
        field_domain: torch.Tensor,
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
            t=t,
        )
        # time embedding (batch_size, embed_dim, *field_dims)
        t_embed = self._get_time_embedding(t=t, field_dims=field_dims)
        # field domain (batch_size, field_ndim, *field_dims)
        x_domain = field_domain.expand(batch_size, -1, *field_dims)
        # input field
        inp_vt = torch.cat((psi, t_embed, condition), dim=1)
        # compute the vector field
        vt = self.field(x=inp_vt, x_domain=x_domain, cond=condition)
        return vt
