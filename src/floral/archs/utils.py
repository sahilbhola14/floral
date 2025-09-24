# src/archs/utils.py
"""
contains utilities for building architectures
"""
import torch.nn as nn


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


class LayerNorm32(nn.LayerNorm):
    """Layer normalization for 1D inputs with arbitrary precision"""

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
