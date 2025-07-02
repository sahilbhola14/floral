import torch.nn as nn


class MLP(nn.Module):
    """Class that implements a multi-Layer Perceptron (MLP) class."""

    def __init__(
        self,
        in_dim=1,
        width=[32, 32, 32],
        out_dim=1,
        activations=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
    ):
        """
        Args:
            in_dim (int): Input dimension of the MLP.
            width (list): List of integers representing the width of each layer.
            out_dim (int): Output dimension of the MLP.
            activations (list): List of activation functions to be applied.
        """
        super(MLP, self).__init__()
        """ init MLP """
        assert (
            len(width) == len(activations) - 1
        ), "Output activation must also be specified (None for no activation)"
        layers = []
        for ii, jj in zip([in_dim] + width[:-1], width):
            layers.extend([nn.Linear(ii, jj), activations.pop(0)])
        layers.append(nn.Linear(width[-1], out_dim))
        if activations[0] is not None:
            layers.append(activations[0])
        self.net = nn.Sequential(*layers)

    def forward(self, input):
        """forward pass"""
        return self.net(input)
