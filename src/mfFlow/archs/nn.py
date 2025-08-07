import torch
import torch.nn as nn
from .encoding import FiLM


def zero_module(module: nn.Module):
    """zero out the parameters of a module and return it"""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization1D(in_features):
    return LayerNorm32(in_features)


class LayerNorm32(nn.LayerNorm):
    """Layer normalization for 1D inputs with arbitrary precision"""

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class Res1DBlock(nn.Module):
    """Residual block for 1D inputs"""

    def __init__(self, in_features, time_emb_dim: int, hidden_dims: list, **kwargs):
        super(Res1DBlock, self).__init__()
        self.in_features = in_features
        self.time_emb_dim = time_emb_dim
        self.hidden_dims = hidden_dims

        # skip connection
        self.skip_connection = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
        )

        # embedding layer
        self.emb_layers = nn.Sequential(
            normalization1D(self.in_features),
            nn.Linear(self.in_features, self.in_features),
            nn.SiLU(),
            nn.Linear(self.in_features, self.hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(kwargs.get("dropout", 0.1)),
            nn.Linear(self.hidden_dims[1], self.in_features),
        )

        # output layer
        self.output_layers = zero_module(
            nn.Sequential(
                normalization1D(self.in_features),
                nn.Linear(self.in_features, self.in_features),
                nn.SiLU(),
                nn.Linear(self.in_features, self.in_features),
            )
        )

        # FiLM layer
        self.film_layer = FiLM(self.time_emb_dim, self.in_features)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        """forward pass of the residual block"""
        B, C = x.shape
        # skip
        skip = self.skip_connection(x)
        # embedding
        out = self.emb_layers(x)
        # apply FiLM layer for time embedding
        out = self.film_layer(out.unsqueeze(-1).unsqueeze(-1), time_emb).view(B, C)
        # output
        out = self.output_layers(out)
        # add skip connection
        out = out + skip
        return out


class Conditional1DEmbedding(nn.Module):
    """Class for 1D conditional embedding

    This class embeds an input of (batch_size, nc) into (batch_size, latent_dim)

    Attibutes:
        nc (int): Dimensionality of the conditional input
        latent_dim (int): Dimensionality of the encodedlatent space
        time_emb_freq (int): Frequency of the time embedding
        hidden_dims (list): List of hidden layer sizes
        dropout (float): Dropout rate
        num_res_blocks (int): Number of residual blocks
    """

    def __init__(self, nc: int, latent_dim: int, time_embed_freq: int, **kwargs):
        super(Conditional1DEmbedding, self).__init__()
        self.nc = nc
        self.latent_dim = latent_dim
        self.time_embed_freq = time_embed_freq
        self.hidden_dims = kwargs.get("hiddden_dims", [64, 64])
        self.dropout = kwargs.get("dropout", 0.1)
        self.num_res_blocks = kwargs.get("num_res_blocks", 2)
        self.use_attention = kwargs.get("use_attention", False)

        # time embedding for encoding the time embedding for the conditional input
        self.time_embed_dim = (self.time_embed_freq * 2) * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_freq * 2, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # opening layer to bring the inputs to the latent dim
        if self.nc != self.latent_dim:
            self.opening = nn.Sequential(
                normalization1D(self.nc),
                nn.Linear(self.nc, self.latent_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        else:
            self.opening = nn.Identity()

        # Residual 1D blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(
                Res1DBlock(self.latent_dim, self.time_embed_dim, self.hidden_dims)
            )

    def forward(self, condition: torch.Tensor, time_emb: torch.Tensor):
        """forward pass of the conditional embedding

        Args:
            condition (torch.Tensor): Input tensor of shape (batch_size, nc)
            time_emb (torch.Tensor): Time embedding tensor of shape
            (batch_size, time_embed_freq * 2)

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, latent_dim)
        """
        # time embedding
        time_emb = self.time_embed(time_emb)

        # opening layer
        out = self.opening(condition)

        # residual blocks
        for res_block in self.res_blocks:
            out = res_block(out, time_emb)
        return out


class StateEmbedding(nn.Module):
    """Class for embedding state information
    This class embeds an input of (batch_size, nc) into (batch_size, latent_dim)

    Attibutes:
        nx (int): Dimensionality of the state input
        latent_dim (int): Dimensionality of the encodedlatent space
        time_emb_freq (int): Frequency of the time embedding
        hidden_dims (list): List of hidden layer sizes
        dropout (float): Dropout rate
        num_res_blocks (int): Number of residual blocks
    """

    def __init__(self, nx: int, latent_dim: int, time_embed_freq: int, **kwargs):
        super(StateEmbedding, self).__init__()
        self.nx = nx
        self.latent_dim = latent_dim
        self.time_embed_freq = time_embed_freq
        self.hidden_dims = kwargs.get("hiddden_dims", [64, 64])
        self.dropout = kwargs.get("dropout", 0.1)
        self.num_res_blocks = kwargs.get("num_res_blocks", 2)

        # time embedding for encoding the time embedding for the state input
        self.time_embed_dim = (self.time_embed_freq * 2) * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_freq * 2, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # opening layer to bring the inputs to the latent dim
        if self.nx != self.latent_dim:
            self.opening = nn.Sequential(
                normalization1D(self.nx),
                nn.Linear(self.nx, self.latent_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        else:
            self.opening = nn.Identity()

        # Residual 1D blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(
                Res1DBlock(self.latent_dim, self.time_embed_dim, self.hidden_dims)
            )

    def forward(self, state: torch.Tensor, time_emb: torch.Tensor):
        """forward pass of the state embedding

        Args:
            state (torch.Tensor): Input tensor of shape (batch_size, nx)
            time_emb (torch.Tensor): Time embedding tensor of shape
            (batch_size, time_embed_freq * 2)

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, latent_dim)
        """
        # time embedding
        time_emb = self.time_embed(time_emb)

        # opening layer
        out = self.opening(state)

        # residual blocks
        for res_block in self.res_blocks:
            out = res_block(out, time_emb)
        return out
