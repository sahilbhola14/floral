# src/floral/archs/operators.py
"""
operator modules
adapted from https://github.com/yzshi5/SPL_OFM/blob/main/models/fno.py
modified to account for fourier time embedding,
conditional inputs, and domain embedding.
"""
import torch
import torch.nn as nn
from collections.abc import Mapping
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf

from neuralop.models import FNO as _FNO
from floral.utils import check_keys, check_tensor_blowup, printer
from .embedding import build_domain_encoder
from .fno import FiLMFNO


@dataclass
class VectorFieldConfig:
    """Default config for VectorField with override support."""

    config: dict = field(
        default_factory=lambda: {
            "method": "FiLMFNO",
            "field": {
                "channels": 1,
                "ndim": 1,
                "hidden_channels": 64,
                "lifting_channel_ratio": 4,
                "projection_channel_ratio": 4,
                "n_layers": 4,
                "modes": 64,
                "linear_skip": True,
            },
            "condition": {
                "channels": 1,
                "ndim": 1,
            },
        }
    )

    @staticmethod
    def _to_dict_config(value):
        if value is None:
            return OmegaConf.create({})
        if isinstance(value, DictConfig):
            return value
        if isinstance(value, Mapping):
            return OmegaConf.create(dict(value))
        if hasattr(value, "items"):
            return OmegaConf.create({k: v for k, v in value.items()})
        raise TypeError(f"Unsupported config type: {type(value).__name__}")

    def resolve(self, config=None):
        return OmegaConf.merge(
            OmegaConf.create(self.config),
            self._to_dict_config(config),
        )


def get_vector_field_operator(operator_config: dict, floral: bool = False):
    """build the operator modules
    Args:
        operator_config (dict):
            configuration for building the operator
    """
    return VectorField(config=operator_config, floral=floral)


class VectorField(nn.Module):
    def __init__(
        self,
        config: dict,
        floral: bool = False,
    ):
        super(VectorField, self).__init__()
        # resolve config against defaults
        self.config = VectorFieldConfig().resolve(config=config)
        field_config = self.config.field
        condition_config = self.config.condition

        # check required keys
        check_keys(field_config, ["channels", "ndim"])
        check_keys(condition_config, ["channels", "ndim"])

        # set attributes
        self.method = self.config.method
        self.floral = floral
        self.field_channels = field_config.channels
        self.field_ndim = field_config.ndim
        self.field_hidden_channels = field_config.hidden_channels
        self.field_lifting_channel_ratio = field_config.lifting_channel_ratio
        self.field_projection_channel_ratio = field_config.projection_channel_ratio
        self.field_n_layers = field_config.n_layers
        self.field_modes = field_config.modes
        self.field_linear_skip = field_config.linear_skip
        self.condition_channels = condition_config.channels
        self.condition_ndim = condition_config.ndim

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
        # field channels
        field_channels = self.field_channels
        # in_channels to the field
        lf_channels = self.field_channels if self.floral else 0
        if self.method == "FiLMFNO":
            """FiLM is used to condition"""
            in_channels = field_channels + self.time_embed_dim
            # FiLMFNO
            field_model = FiLMFNO(
                in_channels=in_channels,
                out_channels=self.field_channels,
                hidden_channels=self.field_hidden_channels,
                lifting_channel_ratio=self.field_lifting_channel_ratio,
                projection_channel_ratio=self.field_projection_channel_ratio,
                n_modes=n_modes,
                cond_channels=self.condition_channels + lf_channels,
                linear_skip=self.field_linear_skip,
            )
        elif self.method == "FNO":
            """Regular FNO is used where conditions are concatenated"""
            in_channels = (
                field_channels
                + self.time_embed_dim
                + self.condition_channels
                + lf_channels
            )
            field_model = _FNO(
                in_channels=in_channels,
                out_channels=self.field_channels,
                hidden_channels=self.field_hidden_channels,
                lifting_channel_ratio=self.field_lifting_channel_ratio,
                projection_channel_ratio=self.field_projection_channel_ratio,
                n_modes=n_modes,
                n_layers=self.field_n_layers,
            )
        else:
            raise ValueError(f"Invalid FNO operator: {self.method} selected")

        printer(f"Using {self.method} operator")
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
        assert condition.shape[1] == self.condition_channels, (
            f"expected {self.condition_channels} condition channels, "
            f" got {condition.shape[1]}"
        )
        assert (
            t.ndim == 2 and t.shape[1] == 1
        ), f"expected time of shape (batch_size, 1), got {t.shape}"

        assert field_domain.ndim == self.field_ndim + 2, (
            "expected field domain of shape "
            f"(batch_size, {self.field_ndim}, {field.shape[2:]})"
        )

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

    def _get_operator_input(
        self,
        psi: torch.Tensor,
        t_embed: torch.Tensor,
        condition: torch.Tensor,
        LF_field: torch.Tensor,
    ):
        """input to the operator"""
        # psi_mod = torch.cat((psi, LF_field), dim=1) if self.floral else psi
        psi_mod = psi
        if self.method == "FiLMFNO":
            input_vt = torch.cat((psi_mod, t_embed), dim=1)
        elif self.method == "FNO":
            input_vt = torch.cat((psi_mod, t_embed, condition), dim=1)
        else:
            raise ValueError(f"Method: {self.method} input does not exist")
        return input_vt

    def forward(
        self,
        psi: torch.Tensor,
        condition: torch.Tensor,
        LF_field: torch.Tensor,
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
        # augment condition with LF_field when in floral mode
        cond_input = (
            torch.cat((condition, LF_field), dim=1) if self.floral else condition
        )
        # time embedding (batch_size, embed_dim, *field_dims)
        t_embed = self._get_time_embedding(t=t, field_dims=field_dims)
        # get the input
        input_vt = self._get_operator_input(
            psi=psi,
            t_embed=t_embed,
            condition=cond_input,
            LF_field=LF_field,
        )
        # compute the vector field
        if self.method == "FiLMFNO":
            vt = self.field(x=input_vt, cond=cond_input)
        elif self.method == "FNO":
            vt = self.field(input_vt)
        else:
            raise ValueError(f"Method: {self.method} vector field does not exist")
        check_tensor_blowup(vt, name="vector field")
        return vt
