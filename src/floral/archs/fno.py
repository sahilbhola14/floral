"""
FiLM FNO for conditioning.
Modified from:https://neuraloperator.github.io/dev/_modules/neuralop/models/fno.html
Author: Sahil Bhola, University of Michigan, Ann Arbor, 2025
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
from neuralop.models.base_model import BaseModel
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.embeddings import GridEmbeddingND, GridEmbedding2D
from neuralop.layers.padding import DomainPadding
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.complex import ComplexValued
from typing import Tuple, List, Union, Literal

warnings.filterwarnings("once", category=UserWarning)


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

        # Positional embedding
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(
                    f"Error: expected {self.n_dim}-d positional embeddings, "
                    f"got {positional_embedding}"
                )
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            raise ValueError(
                "Error: tried to instantiate FNO positional embedding with "
                f"{positional_embedding} expected one of 'grid', GridEmbeddingND"
            )

        # Domain padding
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        # Resolution scaling factor
        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        # Lifting module
        self.lifting_module = self._get_lifting_module()
        # FNO module
        self.fno_module = self._get_fno_module()
        # Projection module
        self.projection_module = self._get_projection_module()

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

    def forward(self, x, output_shape=None, cond=None, **kwargs):
        if kwargs:
            warnings.warn(
                f"FNO.forward() received unexpected keyword arguments: "
                f"{list(kwargs.keys())}. These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]
        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        # lifting (B, hidden_channels, *dims)
        x = self.lifting_module(x)
        # domain padding
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        # fno blocks
        for layer_idx in range(self.n_layers):
            x = self.fno_module(x, layer_idx, output_shape=output_shape[layer_idx])
        # domain padding
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        # projection
        x = self.projection_module(x)
        return x


if __name__ == "__main__":
    sample = torch.randn(1, 1, 10)
    fno = FiLMFNO(n_modes=(32,), in_channels=1, out_channels=1, hidden_channels=32)
    fno.forward(sample)
