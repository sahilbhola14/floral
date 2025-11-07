# src/floral/archs/operators.py
"""
operator modules
adapted from https://github.com/yzshi5/SPL_OFM/blob/main/models/fno.py
modified to account for fourier time embedding,
conditional inputs, and domain embedding.
"""
import torch
import torch.nn as nn

from neuralop.models import FNO as _FNO
from floral.utils import check_keys, check_tensor_blowup, printer
from .embedding import build_domain_encoder
from .fno import FiLMFNO


def get_vector_field_operator(operator_config: dict):
    """build the operator modules
    Args:
        operator_config (dict):
            configuration for building the operator
    """
    # check the required keys
    required_keys = ["method", "field", "condition"]
    check_keys(operator_config, required_keys)
    # create model
    return VectorField(
        field_config=operator_config["field"],
        condition_config=operator_config["condition"],
        method=operator_config["method"],
    )


class VectorField(nn.Module):
    def __init__(
        self,
        field_config: dict,
        condition_config: dict,
        method: str = "FiLMFNO",
        **kwargs,
    ):
        super(VectorField, self).__init__()
        # intial check
        required_keys = ["channels", "ndim"]
        check_keys(field_config, required_keys)
        check_keys(condition_config, required_keys)
        # set field attributes
        self.method = method
        self.field_channels = field_config.get("channels")
        self.field_ndim = field_config.get("ndim")
        self.field_hidden_channels = field_config.get("hidden_channels", 128)
        self.field_lifting_channel_ratio = field_config.get("lifting_channel_ratio", 2)
        self.field_projection_channel_ratio = field_config.get(
            "projection_channel_ratio", 2
        )
        self.field_n_layers = field_config.get("n_layers", 4)
        self.field_modes = field_config.get("modes", 32)
        # set condition attributes
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
        if self.method == "FiLMFNO":
            in_channels = self.field_channels + self.time_embed_dim
            # FiLMFNO
            field_model = FiLMFNO(
                in_channels=in_channels,
                out_channels=self.field_channels,
                hidden_channels=self.field_hidden_channels,
                lifting_channel_ratio=self.field_lifting_channel_ratio,
                projection_channel_ratio=self.field_projection_channel_ratio,
                n_modes=n_modes,
                cond_channels=self.condition_channels,
            )
        elif self.method == "FNO":
            in_channels = (
                self.field_channels + self.time_embed_dim + self.condition_channels
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
        # compute the vector field
        if self.method == "FiLMFNO":
            inp_vt = torch.cat((psi, t_embed), dim=1)
            vt = self.field(x=inp_vt, cond=condition)
        elif self.method == "FNO":
            inp_vt = torch.cat((psi, t_embed, condition), dim=1)
            vt = self.field(inp_vt)
        check_tensor_blowup(vt, name="vector field")
        return vt
