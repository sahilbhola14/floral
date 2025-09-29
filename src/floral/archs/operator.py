# src/floral/archs/operator
""" contains operator class """
import torch
from neuralop.models import FNO as _FNO
from .embedding import SpatialFiLM, AttentionPool


def get_operator_modules(
    operator_params: dict,
    time_embed_dim: int,
):
    """get the operator modules for the field and condition
    Args:
        operator_params (Dict[Dict[int]):
            Dictionary of the operator parameters
        time_embed_dim (int):
            time embedding dimension
    """

    def _check_keys(operator_params):
        required_params = ["dims", "ch_in", "ch_out", "ch_hidden", "n_modes"]
        required_keys = ["field", "condition"]
        missing_params = [k for k in required_params if k not in operator_params]
        if len(missing_params) > 0:
            print(f"missing params: {', '.join(missing_params)}")
        for param in operator_params:
            assert all(
                [k in operator_params.get(param).keys() for k in required_keys]
            ), f"{param} mising field/condition keys"
            assert isinstance(
                operator_params.get(param).get("field"), int
            ), f"{param}['field'] should have integer values only"
            assert isinstance(
                operator_params.get(param).get("condition"), int
            ), f"{param}['condition'] should have integer values only"

    # check keys
    _check_keys(operator_params)
    assert time_embed_dim > 0 and isinstance(
        time_embed_dim, int
    ), "time embed dim must be positive integer"

    # condition operator
    condition_operator = ConditionOperator(
        condition_dims=operator_params.get("dims").get("condition"),
        condition_ch_in=operator_params.get("ch_in").get("condition"),
        condition_ch_out=operator_params.get("ch_out").get("condition"),
        condition_ch_hidden=operator_params.get("ch_hidden").get("condition"),
        condition_n_modes=operator_params.get("n_modes").get("condition"),
        time_embed_dim=time_embed_dim,
    )

    # field operator
    field_operator = FieldOperator(
        field_dims=operator_params.get("dims").get("field"),
        field_ch_in=operator_params.get("ch_in").get("field"),
        field_ch_out=operator_params.get("ch_out").get("field"),
        field_ch_hidden=operator_params.get("ch_hidden").get("field"),
        field_n_modes=operator_params.get("n_modes").get("field"),
        condition_ch_out=operator_params.get("ch_out").get("condition"),
        time_embed_dim=time_embed_dim,
    )

    # operator modules
    operator_modules = {
        "field_operator": field_operator,
        "condition_operator": condition_operator,
    }

    return operator_modules


class ConditionOperator(torch.nn.Module):
    """a neural operator for embedding the conditon into a latent space
    Attributes:
        condition_dims(int):
            dimensionality of the condition field,
            e.g., condition_dims=2 for 2d condition
        condition_ch_in (int):
            input channels for the condition of shape
            (B, condition_ch_in, *condition_grid)
        condition_ch_out (int):
            output channels for the embedded condition of shape (B, condition_ch_out)
        condition_ch_hidden (int):
            hidden channels in the FNO
        condition_n_modes (int):
            number of modes (in each direction) for the FNO
        time_embed_dim (int):
            time embedding dimension
    """

    def __init__(
        self,
        condition_dims: int,
        condition_ch_in: int,
        condition_ch_out: int,
        condition_ch_hidden: int,
        condition_n_modes: int,
        time_embed_dim: int,
    ):
        super(ConditionOperator, self).__init__()
        self.condition_dims = condition_dims
        self.condition_ch_in = condition_ch_in
        self.condition_ch_out = condition_ch_out
        self.time_embed_dim = time_embed_dim

        # domain film (modulate spatially)
        self.domain_film = SpatialFiLM(
            mod_ch=self.condition_dims,
            target_ch=self.condition_ch_in,
        )

        # fno model
        self.model = _FNO(
            n_modes=(condition_n_modes,) * self.condition_dims,
            in_channels=self.condition_ch_in + self.time_embed_dim,
            out_channels=self.condition_ch_out,
            hidden_channels=condition_ch_hidden,
        )

        # attention pool
        self.pool = AttentionPool(in_channels=self.condition_ch_out)

    def _check_input(
        self,
        condition: torch.Tensor,
        condition_domain: torch.Tensor,
        time_embed: torch.Tensor,
    ):
        """check the inputs"""
        # check condition
        _, condition_ch_in, *condition_grid = condition.shape
        assert (
            condition_ch_in == self.condition_ch_in
        ), f"expected {self.condition_ch_in} channels, got {condition_ch_in}"
        assert (
            len(condition_grid) == self.condition_dims
        ), f"expected {self.condition_dims}D grid, got {len(condition_grid)}"
        # check domain
        _, condition_domain_ch_in, *condition_domain_grid = condition_domain.shape
        assert (
            condition_domain_ch_in == self.condition_dims
        ), f"incorrect domain channels. expected {self.condition_domain},"
        f"got {condition_domain_ch_in}"
        assert all(
            [ii == jj for ii, jj in zip(condition_grid, condition_domain_grid)]
        ), "incorrect domain grid. expected {condition_grid},"
        f"got {condition_domain_grid}"
        # check time embed
        assert (time_embed.ndim == 2) and isinstance(
            time_embed, torch.Tensor
        ), "time embedding must be 2D tensor"
        _, time_embed_dim = time_embed.shape
        assert (
            time_embed_dim == self.time_embed_dim
        ), f"expected {self.time_embed_dim}, got {time_embed_dim}"

    def forward(
        self,
        condition: torch.Tensor,
        condition_domain: torch.Tensor,
        time_embed: torch.Tensor,
    ):
        """
        Args:
            condition (torch.Tensor) :
                condition for the pde of shape (B, condition_ch_in, *cond_grid)
            condition_domain (torch.Tensor):
                domain for the conditon of shape (1, condition_dims, *cond_grid)
            time_embed (torch.Tensor):
                time embedding of shape (B, time_embed_dim) to apply to the condition.

        Returns:
            condition_embed (torch.Tensor):
                condition embedding of shape (B, cond_ch_out) with attnetion pooling to
                retain global spatial context for the condition.
        """
        # check the input
        self._check_input(condition, condition_domain, time_embed)
        # film on the domain
        B, condition_ch_in, *condition_grid = condition.shape
        condition = self.domain_film(target=condition, mod=condition_domain)
        # concatenate time to the condition (channel-wise)
        time_expanded = time_embed.view(
            B, self.time_embed_dim, *([1] * len(condition_grid))
        ).expand(-1, -1, *condition_grid)
        inp = torch.cat([condition, time_expanded], dim=1)
        # pass through operator
        output = self.model(inp)
        B, condition_ch_out, *condition_out_grid = output.shape
        assert (
            condition_ch_out == self.condition_ch_out
        ), "incorrect condition operator output channels"
        assert all(
            [ii == jj for ii, jj in zip(condition_grid, condition_out_grid)]
        ), "in correct condition operator output grid"
        # attention pooling
        output = self.pool(output)
        return output


class FieldOperator(torch.nn.Module):
    """a neural operator for embedding the field
    Attributes:
        field_dims(int):
            dimensionality of the field field, e.g., field_dims=2 for 2d field
        field_ch_in (int):
            input channels for the field of shape (B, field_ch_in, *field_grid)
        field_ch_out (int):
            output channels for the embedded field of shape (B, field_ch_out)
        field_ch_hidden (int):
            hidden channels in the FNO
        field_n_modes (int):
            number of modes (in each direction) for the FNO
        condition_ch_out (int):
            output channels for the embedded condition of shape (B, condition_ch_out)
        time_embed_dim (int):
            time embedding dimension
    """

    def __init__(
        self,
        field_dims: int,
        field_ch_in: int,
        field_ch_out: int,
        field_ch_hidden: int,
        field_n_modes: int,
        condition_ch_out: int,
        time_embed_dim: int,
    ):
        super(FieldOperator, self).__init__()
        self.field_dims = field_dims
        self.field_ch_in = field_ch_in
        self.field_ch_out = field_ch_out
        self.condition_ch_out = condition_ch_out
        self.time_embed_dim = time_embed_dim

        # domain film (modulate spatially)
        self.domain_film = SpatialFiLM(
            mod_ch=self.field_dims,
            target_ch=self.field_ch_in,
        )

        # fno model
        self.model = _FNO(
            n_modes=(field_n_modes,) * self.field_dims,
            in_channels=self.field_ch_in + self.time_embed_dim + self.condition_ch_out,
            out_channels=self.field_ch_out,
            hidden_channels=field_ch_hidden,
        )

    def _check_input(
        self,
        field: torch.Tensor,
        field_domain: torch.Tensor,
        condition_embed: torch.Tensor,
        time_embed: torch.Tensor,
    ):
        """check the inputs"""
        # check field
        _, field_ch_in, *field_grid = field.shape
        assert (
            field_ch_in == self.field_ch_in
        ), f"expected {self.field_ch_in} channels, got {field_ch_in}"
        assert (
            len(field_grid) == self.field_dims
        ), f"expected {self.field_dims}D grid, got {len(field_grid)}"
        # check domain
        _, field_domain_ch_in, *field_domain_grid = field_domain.shape
        assert (
            field_domain_ch_in == self.field_dims
        ), f"incorrect domain channels. expected {self.field_domain},"
        f"got {field_domain_ch_in}"
        assert all(
            [ii == jj for ii, jj in zip(field_grid, field_domain_grid)]
        ), "incorrect domain grid. expected {field_grid},"
        f"got {field_domain_grid}"
        # check condition embed
        assert (condition_embed.ndim == 2) and isinstance(
            condition_embed, torch.Tensor
        ), "condition embedding must be 2D tensor"
        (
            _,
            condition_ch_in,
        ) = condition_embed.shape
        assert (
            condition_ch_in == self.condition_ch_out
        ), "expected condition embedding to be of shape"
        f"(batch_size, {self.condition_ch_out}, got (condition_embed.shape))"
        # check time embed
        assert (time_embed.ndim == 2) and isinstance(
            time_embed, torch.Tensor
        ), "time embedding must be 2D tensor"
        _, time_embed_dim = time_embed.shape
        assert (
            time_embed_dim == self.time_embed_dim
        ), f"expected {self.time_embed_dim}, got {time_embed_dim}"

    def forward(
        self,
        field: torch.Tensor,
        field_domain: torch.Tensor,
        condition_embed: torch.Tensor,
        time_embed: torch.Tensor,
    ):
        """
        Args:
            field (torch.Tensor) :
                field for the pde of shape (B, field_ch_in, *cond_grid)
            field_domain (torch.Tensor):
                domain for the conditon of shape (1, field_dims, *cond_grid)
            condition_embed (torch.Tensor):
                time embedding of shape (B, condition_ch_out) to apply to the field.
            time_embed (torch.Tensor):
                time embedding of shape (B, time_embed_dim) to apply to the field.

        Returns:
            field_embed (torch.Tensor):
                field embedding of shape (B, cond_ch_out) with attnetion pooling to
                retain global spatial context for the field.
        """
        # check the input
        self._check_input(field, field_domain, condition_embed, time_embed)
        # film on the domain (B, field_ch_in, *field_grid)
        B, field_ch_in, *field_grid = field.shape
        field = self.domain_film(target=field, mod=field_domain)
        # concatenate time to the field (channel-wise)
        time_expanded = time_embed.view(
            B, self.time_embed_dim, *([1] * len(field_grid))
        ).expand(-1, -1, *field_grid)
        condition_expanded = condition_embed.view(
            B, self.condition_ch_out, *([1] * len(field_grid))
        ).expand(-1, -1, *field_grid)
        inp = torch.cat(
            [field, time_expanded, condition_expanded], dim=1
        )  # (B, field_ch_in + time_embed_dim + condition_ch_out, *field_grid)
        assert (
            inp.shape[1]
            == self.field_ch_in + self.time_embed_dim + self.condition_ch_out
        ), "incorrect channels, expected"
        f"{self.field_ch_in + self.time_embed_dim + self.condition_ch_out},"
        f"got {inp.shape[1]}"
        # pass through operator
        output = self.model(inp)
        B, field_ch_out, *field_out_grid = output.shape
        assert (
            field_ch_out == self.field_ch_out
        ), "incorrect field operator output channels"
        assert all(
            [ii == jj for ii, jj in zip(field_grid, field_out_grid)]
        ), "in correct field operator output grid"
        return output
