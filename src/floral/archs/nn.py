# import math
# import torch
import torch.nn as nn

# from .encoding import (
#     FiLM,
#     MLP,
#     RBFFiLM,
#     conv_nd,
#     SpatialAttentionPooling,
#     SpatialAdaptivePooling,
#     CoordModulation,
# )


def zero_module(module: nn.Module):
    """zero out the parameters of a module and return it"""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization1D(in_features):
    """Layer Normalize"""
    return LayerNorm32(in_features)


def normalization2D(channels):
    """Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# def get_embedding_modules(
#     nx: int,
#     nc: int,
#     nd: int,
#     nd_c: int,
#     latent_dim: int,
#     time_embed_freq: int,
#     num_centers: int,
#     field_data: bool = False,
#     condition_domain: torch.Tensor = None,
#     **kwargs,
# ):
#     """Wrapper to get the modules for StateEmbedding, ConditionEmbedding,
#     FusionEmbedding, and DomainEmbedding"""
#     # state embedding
#     state_embedding = StateEmbedding(
#         nx=nx,
#         latent_dim=latent_dim,
#         time_embed_freq=time_embed_freq,
#         **kwargs,
#     )
#     # condition embedding
#     if field_data:
#         condition_embedding = Condition2DEmbedding(
#             nc=nc,
#             nd_c=nd_c,
#             latent_dim=latent_dim,
#             time_embed_freq=time_embed_freq,
#             condition_domain=condition_domain,
#             **kwargs,
#         )
#     else:
#         condition_embedding = Condition1DEmbedding(
#             nc=nc,
#             nd_c=nd_c,
#             latent_dim=latent_dim,
#             time_embed_freq=time_embed_freq,
#             condition_domain=condition_domain,
#             **kwargs,
#         )
#     # fusion embedding
#     fusion_embedding = FusionEmbedding(
#         latent_dim=latent_dim,
#         time_embed_freq=time_embed_freq,
#         **kwargs,
#     )

#     # domain embedding
#     domain_embedding = RBFFiLM(
#         num_centers=num_centers,
#         latent_dim=latent_dim,
#         nd=nd,
#         nx=nx,
#         **kwargs,
#     )

#     embedding = {
#         "state_embedding": state_embedding,
#         "condition_embedding": condition_embedding,
#         "fusion_embedding": fusion_embedding,
#         "domain_embedding": domain_embedding,
#     }
#     return embedding


class LayerNorm32(nn.LayerNorm):
    """Layer normalization for 1D inputs with arbitrary precision"""

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
