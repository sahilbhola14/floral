"""
FiLM FNO for conditioning.
Modified from:https://neuraloperator.github.io/dev/_modules/neuralop/models/fno.html
Author: Sahil Bhola, University of Michigan, Ann Arbor, 2025
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import warnings
from torch.utils.data import TensorDataset, DataLoader
from neuralop.models.base_model import BaseModel
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.embeddings import GridEmbeddingND, GridEmbedding2D
from neuralop.layers.padding import DomainPadding
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.complex import ComplexValued
from typing import Tuple, List, Union, Literal, Optional
from .embedding import ChannelFiLM, conv_nd

warnings.filterwarnings("once", category=UserWarning)


class LinearOperator(nn.Module):
    """Global linear operator: spectral conv (K) + pointwise conv (W).

    Mirrors the K + W decomposition from the FNO paper applied as a single
    input-to-output bypass:
      - SpectralConv captures global spatial correlations via Fourier modes
      - pointwise Conv1x1 handles local channel mixing
    Both branches are linear; no activations are applied, so the sum remains
    a linear operator. Lightweight: only one extra 1x1 conv over the bare
    spectral version.
    """

    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
    ):
        super(LinearOperator, self).__init__()
        n_dim = len(n_modes)
        self.K = SpectralConv(in_channels, out_channels, n_modes)
        self.W = conv_nd(n_dim, in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        return self.K(x) + self.W(x)


class FiLMFNO(BaseModel, name="FiLMFNO"):
    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: Union[float, int] = 2,
        projection_channel_ratio: Union[float, int] = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        cond_positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal[
            "linear", "identity", "soft-gating", None
        ] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Union[
            Union[float, int], List[Union[float, int]]
        ] = None,
        domain_padding: Union[Union[float, int], List[Union[float, int]]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        cond_channels: Optional[int] = None,
        linear_skip: bool = True,
    ):

        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__()
        self.n_dim = len(n_modes)

        # n_modes is a special property
        # When updated, change should be reflected in fno blocks
        self._n_modes = n_modes

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # init lifting and projection channels using ratios w.r.t hidden channels
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int(lifting_channel_ratio * self.hidden_channels)

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * self.hidden_channels)

        self.non_linearity = non_linearity
        self.rank = rank
        self.use_channel_mlp = use_channel_mlp
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.channel_mlp_dropout = channel_mlp_dropout
        self.channel_mlp_expansion = channel_mlp_expansion
        self.stabilizer = stabilizer
        self.norm = norm
        self.max_n_modes = max_n_modes
        self.conv_module = conv_module
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.complex_data = complex_data
        self.fno_block_precision = fno_block_precision
        self.cond_channels = cond_channels  # number of channels in the condition
        self.apply_cond = self.cond_channels is not None
        self.linear_skip = linear_skip

        # positional embedding
        self.positional_embedding = self._get_positional_embedding(
            positional_embedding, self.in_channels, self.n_dim
        )
        # domain padding
        self.domain_padding = self._get_domain_padding(
            domain_padding, resolution_scaling_factor
        )
        # resolution scaling
        self.resolution_scaling_factor = self._get_resolution_scaling_factor(
            resolution_scaling_factor
        )

        # linear operator (optional skip connection)
        if self.linear_skip:
            self.linear_operator = LinearOperator(
                n_modes=self._n_modes,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
            )

        # lifting module
        self.lifting_module = self._get_lifting_module()
        # FNO module
        self.fno_module = self._get_fno_module()
        # projection module
        self.projection_module = self._get_projection_module()
        # condition
        if self.apply_cond:
            # positiona embedding
            self.cond_positional_embedding = self._get_positional_embedding(
                cond_positional_embedding, self.cond_channels, self.n_dim
            )
            # lifting
            self.cond_lifting_module = self._get_cond_lifting_module()
            # film (for linear operator)
            self.cond_linear_op_film = ChannelFiLM(
                target_channels=self.out_channels, embed_features=self.cond_channels
            )
            # film layers
            self.cond_lifting_film = ChannelFiLM(
                target_channels=self.hidden_channels,
                embed_features=self.hidden_channels,
            )
            # FNO FiLM module
            self.cond_fno_film = nn.ModuleList([])
            for layer in range(self.n_layers):
                self.cond_fno_film.append(
                    ChannelFiLM(
                        target_channels=self.hidden_channels,
                        embed_features=self.hidden_channels,
                    )
                )

    def _get_positional_embedding(self, positional_embedding, in_channels, n_dim):
        """get the positional embedding"""
        # Positional embedding
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * n_dim
            embedding = GridEmbeddingND(
                in_channels=in_channels,
                dim=n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            if n_dim == 2:
                embedding = positional_embedding
            else:
                raise ValueError(
                    f"Error: expected {n_dim}-d positional embeddings, "
                    f"got {positional_embedding}"
                )
        elif isinstance(positional_embedding, GridEmbeddingND):
            embedding = positional_embedding
        elif positional_embedding is None:
            embedding = None
        else:
            raise ValueError(
                "Error: tried to instantiate FNO positional embedding with "
                f"{positional_embedding} expected one of 'grid', GridEmbeddingND"
            )
        return embedding

    def _get_domain_padding(self, domain_padding, resolution_scaling_factor):
        """domain padding"""
        # Domain padding
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            padding = None
        return padding

    def _get_resolution_scaling_factor(self, resolution_scaling_factor):
        """get the resolution scaling"""
        # Resolution scaling factor
        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                scaling_factor = [resolution_scaling_factor] * self.n_layers
        else:
            scaling_factor = resolution_scaling_factor
        return scaling_factor

    def _get_lifting_module(self):
        """get the lifting module"""
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        # if lifting_channels is passed, make lifting a Channel-Mixing MLP
        # with a hidden layer of size lifting_channels

        if self.lifting_channels:
            lifting_module = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=self.non_linearity,
            )
        # otherwise, make it a linear layer
        else:
            lifting_module = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=self.non_linearity,
            )
        # Convert lifting to a complex ChannelMLP if self.complex_data==True
        if self.complex_data:
            lifting_module = ComplexValued(self.lifting)
        return lifting_module

    def _get_cond_lifting_module(self):
        """condition lifting"""
        lifting_in_channels = self.cond_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        # if lifting_channels is passed, make lifting a Channel-Mixing MLP
        # with a hidden layer of size lifting_channels

        if self.lifting_channels:
            lifting_module = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=self.non_linearity,
            )
        # otherwise, make it a linear layer
        else:
            lifting_module = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=self.non_linearity,
            )
        # Convert lifting to a complex ChannelMLP if self.complex_data==True
        if self.complex_data:
            lifting_module = ComplexValued(self.lifting)

        return lifting_module

    def _get_fno_module(self):
        """get the FNO module"""
        fno_module = FNOBlocks(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            n_modes=self._n_modes,
            resolution_scaling_factor=self.resolution_scaling_factor,
            use_channel_mlp=self.use_channel_mlp,
            channel_mlp_dropout=self.channel_mlp_dropout,
            channel_mlp_expansion=self.channel_mlp_expansion,
            non_linearity=self.non_linearity,
            stabilizer=self.stabilizer,
            norm=self.norm,
            preactivation=self.preactivation,
            fno_skip=self.fno_skip[0],
            channel_mlp_skip=self.channel_mlp_skip[0],
            complex_data=self.complex_data,
            max_n_modes=self.max_n_modes,
            fno_block_precision=self.fno_block_precision,
            rank=self.rank,
            fixed_rank_modes=self.fixed_rank_modes,
            implementation=self.implementation,
            separable=self.separable,
            factorization=self.factorization,
            decomposition_kwargs=self.decomposition_kwargs,
            conv_module=self.conv_module,
            n_layers=self.n_layers,
        )
        return fno_module

    def _get_projection_module(self):
        """projection module"""
        # Projection layer
        projection_module = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=self.non_linearity,
        )
        if self.complex_data:
            projection_module = ComplexValued(self.projection)
        return projection_module

    def _forward_lifting(self, x, cond=None):
        # state positional embedding
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        # state lifting
        x = self.lifting_module(x)
        # domain padding
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        if self.apply_cond:
            assert (
                self.domain_padding is None
            ), "Domain padding unavailable for condition."
            # condition position embedding
            if self.cond_positional_embedding is not None:
                cond = self.cond_positional_embedding(cond)
            # condition lifting
            cond = self.cond_lifting_module(cond)
            # apply film
            x = self.cond_lifting_film(target=x, embed=cond)

        return x, cond

    def _forward_linear(self, x, cond=None, **kwargs):
        """apply linear operator"""
        # linera operator
        x = self.linear_operator(x=x)
        # film application
        if self.apply_cond is not None:
            x = self.cond_linear_op_film(target=x, embed=cond)
        return x

    def _forward_nonlinear(self, x, cond=None, output_shape=None, **kwargs):
        """apply non-linear operator"""
        # lifting
        x, cond = self._forward_lifting(x=x, cond=cond)
        # fno blocks
        for layer_idx in range(self.n_layers):
            # apply FNO block
            x = self.fno_module(x, layer_idx, output_shape=output_shape[layer_idx])
            # apply FiLM
            x = self.cond_fno_film[layer_idx](target=x, embed=cond)
        # domain padding
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        # projection
        x = self.projection_module(x)

        return x

    def forward(self, x, cond=None, output_shape=None, **kwargs):
        if kwargs:
            warnings.warn(
                f"FNO.forward() received unexpected keyword arguments: "
                f"{list(kwargs.keys())}. These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        assert output_shape is None, "output_shape reshape not currently available"
        if self.apply_cond is not None:
            assert cond is not None, "condition must be provided"
        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]
        # Non-Linear Operator forward
        x_nlin = self._forward_nonlinear(x=x, cond=cond, output_shape=output_shape)
        if self.linear_skip:
            # Linear Operator forward (global spectral skip)
            x_lin = self._forward_linear(x=x, cond=cond)
            return x_lin + x_nlin
        return x_nlin


def test_train():
    """test train"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")
    x = torch.linspace(0, 1, 20)
    n_samples = 3000
    # w(x) = cos(x)
    state = torch.FloatTensor(
        torch.cos(x).unsqueeze(0).expand(n_samples, -1).unsqueeze(1)
    )
    coeffs = [ii / 10 for ii in range(10)]
    # a(x) = coeff * x
    cond = torch.FloatTensor(
        torch.stack([random.choice(coeffs) * x for _ in range(n_samples)]).unsqueeze(1)
    )
    # output
    y = torch.FloatTensor((state * cond).sin())
    # train and val
    n_train = int(0.7 * n_samples)
    state_train, state_val = state[:n_train], state[n_train:]
    cond_train, cond_val = cond[:n_train], cond[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    train_set = TensorDataset(state_train, cond_train, y_train)
    val_set = TensorDataset(state_val, cond_val, y_val)
    train_dataset = DataLoader(train_set, batch_size=128, shuffle=True)
    val_dataset = DataLoader(val_set, batch_size=128, shuffle=True)
    # model
    fno = FiLMFNO(
        n_modes=(32,),
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        cond_channels=1,
    )
    fno = fno.to(device)
    optim = torch.optim.Adam(fno.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    for epoch in range(300):
        # training
        fno.train()
        train_loss = 0.0
        for state, cond, y in train_dataset:
            state = state.to(device)
            cond = cond.to(device)
            y = y.to(device)
            optim.zero_grad()
            # forward pass
            pred = fno.forward(state, cond=cond)
            # loss
            loss = criterion(pred, y)
            loss.backward()
            optim.step()
            train_loss += loss.item() * state.size(0)

        # validation
        fno.eval()
        val_loss = 0.0
        with torch.no_grad():
            for state, cond, y in val_dataset:
                state = state.to(device)
                cond = cond.to(device)
                y = y.to(device)
                pred = fno(state, cond=cond)
                loss = criterion(pred, y)
                val_loss += loss.item() * state.size(0)

        train_loss /= len(train_dataset.dataset)
        val_loss /= len(val_dataset.dataset)

        if epoch % 50 == 0:
            print(
                f"Epoch [{epoch:04d}] | Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

    # testing
    cond_testing = torch.FloatTensor(1.2 * x).view(1, 1, -1).to(device)
    state_testing = torch.FloatTensor(torch.cos(x)).view(1, 1, -1).to(device)
    true = torch.sin(cond_testing * state_testing).to("cpu")
    with torch.no_grad():
        pred = fno(state_testing, cond=cond_testing).to("cpu")
        error = torch.norm(pred - true, p=2)
        print(f"prediction error: {error.item()}")


if __name__ == "__main__":
    # test_train
    test_train()
    # fno = FiLMFNO(n_modes=(32, ),
    #         in_channels=1,
    #         out_channels=1,
    #         hidden_channels=32,
    #         cond_channels=condition.shape[1]
    #         )
    # fno.forward(sample, cond=condition)
