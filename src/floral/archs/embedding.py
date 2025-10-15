# src/floral/archs/embedding.py
"""
Useful embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron with flexible architecture"""

    def __init__(self, in_dim, width, out_dim, activations, dropout=0.0, norm=None):
        super().__init__()
        layers = []
        dims = [in_dim] + width + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Add normalization if specified
            if norm == "layer" and i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
            elif norm == "batch" and i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))

            # Add activation
            if i < len(activations) and activations[i] is not None:
                layers.append(activations[i])

            # Add dropout
            if dropout > 0 and i < len(dims) - 2:
                layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SpatialAttentivePooling(nn.Module):
    """
    Spatial Attentive Pooling to convert (batch_size, in_channels, *dims) to
    (batch_size, in_channels) irrespective of *dims. This is performed to
    make sure there is a (learnable) weighted averaging, rather than simple
    mean.
    """

    def __init__(self, in_channels: int, **kwargs):
        super(SpatialAttentivePooling, self).__init__()
        self.in_channels = in_channels
        activation = nn.SiLU()
        self.net = MLP(
            in_dim=self.in_channels,
            out_dim=self.in_channels,
            activations=[activation, activation, None],
            width=[32, 32],
            dropout=kwargs.get("dropout", 0.0),
        )

    def forward(self, target: torch.Tensor):
        """forwar pass"""
        batch_size, channels, *dims = target.shape
        assert channels == self.in_channels
        # reshape target
        reshape_target = target.flatten(start_dim=2).transpose(1, 2)
        # compute weights
        weights = self.net(reshape_target)
        weights = F.softmax(weights, dim=1)
        # Weighted pooling (elementwise multiply and sum)
        pooled = torch.sum(weights * reshape_target, dim=1)  # (B, C)

        return pooled
