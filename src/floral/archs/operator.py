# src/floral/archs/operator.py
"""
operator modules
adapted from https://github.com/yzshi5/SPL_OFM/blob/main/models/fno.py
modified to account for fourier time embedding and conditional inputs.
"""
import torch
from neuralop.models import FNO as _FNO
from floral.utils import check_keys
from .embedding import SpatialAttentivePooling


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


class ConditionEmbedding(torch.nn.Module):
    def __init__(
        self, condition_config: dict, t_scaling: float = 1.0, pooling: str = "attention"
    ):
        super(ConditionEmbedding, self).__init__()
        # intial check
        required_keys = [
            "hidden_channels",
            "proj_channels",
            "modes",
            "out_channels",
            "channels",
            "ndim",
        ]
        check_keys(condition_config, required_keys)
        # set attributes
        self.t_scaling = t_scaling
        for key in required_keys:
            setattr(self, key, condition_config.get(key))
        # create operator
        n_modes = (self.modes,) * self.ndim
        in_channels = self.channels + self.ndim + 1
        self.fno = _FNO(
            n_modes=n_modes,
            hidden_channels=self.hidden_channels,
            projection_channels=self.proj_channels,
            in_channels=in_channels,
            out_channels=self.out_channels,
        )
        # spatiall attentive pooling
        if pooling == "attention":
            self.pool = SpatialAttentivePooling(in_channels=self.out_channels)
        elif pooling == "mean":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid pooling method: {pooling}")

    def _check_input(
        self, condition: torch.Tensor, condition_domain: torch.Tensor, t: torch.Tensor
    ):
        assert condition.ndim == self.ndim + 2
        assert condition_domain.ndim == 3  # (batch_size, domain_channels, *domain_dims)
        assert t.ndim == 2

    def forward(
        self, condition: torch.Tensor, condition_domain: torch.Tensor, t: torch.Tensor
    ):
        """forward pass"""
        # input check
        self._check_input(condition, condition_domain, t)
        batch_size, _, *dims = condition.shape
        # scale the time
        t = t / self.t_scaling
        # reshape time
        t = t.view(batch_size, 1, *([1] * self.ndim)).expand(-1, -1, *dims)
        # reshape domain
        condition_domain = condition_domain.expand(
            batch_size, *condition_domain.shape[1:]
        )
        # create input
        inp = torch.cat((condition, condition_domain, t), dim=1)
        # compute operator output (batch_size, out_channels, *dims)
        out = self.fno(inp)
        # apply spatial attentive pooling
        out = self.pool(out)

        return out


class VectorField(torch.nn.Module):
    def __init__(
        self,
        field_config: dict,
        condition_config: dict,
        t_scaling: float = 1.0,
        **kwargs,
    ):
        super(VectorField, self).__init__()
        # intial check
        required_keys = [
            "hidden_channels",
            "proj_channels",
            "modes",
            "channels",
            "ndim",
        ]
        check_keys(field_config, required_keys)
        # set attributes
        self.t_scaling = t_scaling
        for key in required_keys:
            setattr(self, key, field_config.get(key))

        # condition FNO
        self.condition_embedding = ConditionEmbedding(condition_config, t_scaling)
        # create operator
        n_modes = (self.modes,) * self.ndim
        in_channels = (
            self.channels + self.ndim + condition_config.get("out_channels") + 1
        )
        self.model = _FNO(
            n_modes=n_modes,
            hidden_channels=self.hidden_channels,
            projection_channels=self.proj_channels,
            in_channels=in_channels,
            out_channels=self.channels,
        )

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
        # extract shape
        batch_size, channels, *dims = psi.shape
        ndim = len(dims)
        # condition embedding (batch_size, condition_out_channels)
        condition_embedding = self.condition_embedding(
            condition=condition, condition_domain=condition_domain, t=t
        )
        condition_embedding = condition_embedding.view(
            batch_size, -1, *([1] * ndim)
        ).expand(-1, -1, *dims)
        # scale the time
        t = t / self.t_scaling
        # reshape time
        t = t.view(batch_size, 1, *([1] * ndim)).expand(-1, -1, *dims)
        # reshape domain
        field_domain = field_domain.expand(batch_size, *field_domain.shape[1:])
        # create input to the FNO
        inp = torch.cat((psi, field_domain, condition_embedding, t), dim=1)
        # compute output
        out = self.model(inp)
        return out
