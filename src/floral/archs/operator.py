# src/floral/archs/operator.py
"""
operator modules
adapted from https://github.com/yzshi5/SPL_OFM/blob/main/models/fno.py
modified to account for fourier time embedding and conditional inputs.
"""
import torch
from neuralop.models import FNO as _FNO
from floral.utils import check_keys


def get_operator_modules(operator_config: dict):
    """build the operator modules
    Args:
        operator_config (dict):
            configuration for building the operator
    """
    # check the required keys
    required_keys = [
        "field_channels",
        "condition_channels",
        "hidden_channels",
        "proj_channels",
        "modes",
        "field_ndim",
    ]
    check_keys(operator_config, required_keys)
    # create model
    return FNO(**operator_config)


def t_allhot(t, shape):
    batch_size = shape[0]
    # n_channels = shape[1]
    dim = shape[2:]
    n_dim = len(dim)

    t = t.view(batch_size, *[1] * (1 + n_dim))
    t = t * torch.ones(batch_size, 1, *dim, device=t.device)
    return t


def conds_allhot(conds, shape):
    # batch_size = shape[0]
    dim = shape[2:]
    n_dim = len(dim)

    # expand conds

    if n_dim == 1:
        conds = conds.unsqueeze(-1)
    elif n_dim == 2:
        conds = conds.unsqueeze(-1).unsqueeze(-1)
    elif n_dim == 3:
        conds = conds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    conds = conds.repeat(1, 1, *dim)
    return conds


def make_posn_embed(batch_size, dims):
    if len(dims) == 1:
        # Single channel of spatial embeddings
        emb = torch.linspace(0, 1, dims[0])
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1)
    elif len(dims) == 2:
        # 2 Channels of spatial embeddings
        x1 = torch.linspace(0, 1, dims[1]).repeat(dims[0], 1).unsqueeze(0)
        x2 = torch.linspace(0, 1, dims[0]).repeat(dims[1], 1).T.unsqueeze(0)
        emb = torch.cat((x1, x2), dim=0)  # (2, dims[0], dims[1])

        # Repeat along new batch channel
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch_size, 2, *dims)

    # new
    elif len(dims) == 3:
        x1 = (
            torch.linspace(0, 1, dims[0])
            .reshape(1, dims[0], 1, 1)
            .repeat(1, 1, dims[1], dims[2])
        )
        x2 = (
            torch.linspace(0, 1, dims[1])
            .reshape(1, 1, dims[1], 1)
            .repeat(1, dims[0], 1, dims[2])
        )
        x3 = (
            torch.linspace(0, 1, dims[2])
            .reshape(1, 1, 1, dims[2])
            .repeat(1, dims[0], dims[1], 1)
        )
        emb = torch.cat((x1, x2, x3), dim=0)

        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # (batch_size, 3, *dims)

    else:
        raise NotImplementedError

    return emb


class FNO(torch.nn.Module):
    def __init__(
        self,
        modes: int,
        field_channels: int,
        condition_channels: int,
        hidden_channels: int,
        proj_channels: int,
        field_ndim: int = 1,
        t_scaling: float = 1.0,
        **kwargs,
    ):
        super(FNO, self).__init__()
        self.t_scaling = t_scaling
        # same modes in each dimension
        n_modes = (modes,) * field_ndim
        # in_channels = (field + domain + t + condition)
        in_channels = field_channels + field_ndim + condition_channels + 1
        # model
        self.model = _FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            projection_channels=proj_channels,
            in_channels=in_channels,
            out_channels=field_channels,
        )

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
                domain of shape (1, domain_channels, *domain_dims)
            t (torch.Tensor):
                samples of time of shape (batch_size, 1)
        """
        # extract shape
        batch_size, field_channels, *field_dims = psi.shape
        field_ndim = len(field_dims)
        # scale the time
        t = t / self.t_scaling
        # reshape time
        t = t.view(batch_size, 1, *([1] * field_ndim)).expand(-1, -1, *field_dims)
        # reshape domain
        field_domain = field_domain.expand(batch_size, *field_domain.shape[1:])
        # create input to the FNO
        inp = torch.cat((psi, field_domain, t, condition), dim=1)
        # compute output
        out = self.model(inp)
        return out
