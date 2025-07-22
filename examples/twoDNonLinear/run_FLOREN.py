# examples/twoDNonLinear/run_FLOREN.py
"""
Flow-Matching Residual Embedded Nerual Opeartor for Two-dimensional problem with
non-linear correlation.
"""
import torch
import torch.nn as nn
import argparse
import math
import pytorch_lightning as L
from omegaconf import OmegaConf
from mfFlow.utils import (
    printer,
    OpDataModule,
    get_checkpointer,
    get_trainer,
    init_weights,
    check_path,
    Inference,
)
from mfFlow.flow import Flow
from mfFlow.archs import RBFFiLM, FiLM

parser = argparse.ArgumentParser(
    description="Run twoDNonLinear with specified parameters."
)
parser.add_argument(
    "--config",
    type=str,
    default="config_FLOREN.yml",
    help="Path to the configuration file.",
)
args = parser.parse_args()
config = OmegaConf.load(args.config)
printer(f"Running twoDNonLinear with configuration: {args.config}")
printer(f"Using mfFlow: {config.mfFlow} and job name: {config.job_name}")
printer(
    f"Number of samples: {config.data.high_fidelity.n_samples} "
    f"sensors: {config.data.high_fidelity.n_sensors}"
)


torch.set_float32_matmul_precision("medium")  # for tensor cores


def get_data_module():
    """Get the data module for the oneDCorr problem."""
    data_module = OpDataModule(
        nx=config.data.nx,
        nc=config.data.nc,
        nd=config.data.nd,
        low_fidelity_path=config.data.low_fidelity.path,
        high_fidelity_path=config.data.high_fidelity.path,
        n_samples=config.data.high_fidelity.n_samples,
        n_sensors=config.data.high_fidelity.n_sensors,
        mfFlow=config.mfFlow,
        dataloader_config=config.dataloader,
        test_data_path=config.data.test_data_path,
    )

    # Setup the data module
    data_module.setup()

    return data_module


class ConvBlock(nn.Module):
    """Class for a convolution block used in condition embedding."""

    def __init__(self, in_channels, out_channels, t_emb_dim):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t_emb_dim = t_emb_dim

        # Opening convolution layer
        self.conv1 = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3, padding=1
        )
        # Second convolution layer
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1
        )

        # FiLM for modulating the output based on time embedding
        self.film = FiLM(2 * self.t_emb_dim, self.out_channels)

        # ReLU
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """Forward pass of the convolution block in U-Net style with Residual
        connection and FiLM modulation.
        """
        # Apply the first convolution layer
        out1 = self.conv1(x)
        # Apply the second convolution layer
        out2 = self.conv2(out1)
        # Apply FiLM modulation
        out2 = self.film(out2, t_emb)
        # Apply ReLU activation
        out2 = self.relu(out2)
        # Add the input to the output (Residual connection)
        out = out1 + out2

        return out


class ConditionEmbedding(nn.Module):
    """Class for condition embedding using a convolution layer."""

    def __init__(self, nc: int, latent_dim: int, time_emb_freq: int):
        """Initialize the condition embedding module.
        Args:
            nc (int): Dimensionality of the condition field.
            latent_dim (int): Dimension of the latent space (output dimension).
            time_emb_freq (int): Frequency of the time embedding.
        """
        super(ConditionEmbedding, self).__init__()
        self.nc = nc
        self.time_emb_freq = time_emb_freq
        self.field_dim = (int(math.sqrt(nc)), int(math.sqrt(nc)))
        assert (
            self.field_dim[0] * self.field_dim[1] == nc
        ), "Condition field must be square."
        self.latent_dim = latent_dim

        # Convolution block for condition embedding
        self.net = nn.ModuleList(
            [
                ConvBlock(1, 32, self.time_emb_freq),
                nn.MaxPool2d(2),  # (H/2, W/2)
                ConvBlock(32, 64, self.time_emb_freq),
                nn.MaxPool2d(2),  # (H/4, W/4)
                ConvBlock(64, 128, self.time_emb_freq),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, self.latent_dim),
            ]
        )

    def forward(self, condition: torch.Tensor, t_emb: torch.Tensor):
        """Forward pass of the condition embedding."""
        # reshape the condition to (batch_size, 1, H, W)
        condition = condition.view(-1, 1, self.field_dim[0], self.field_dim[1])
        # Apply the convolution blocks
        for layer in self.net:
            if isinstance(layer, ConvBlock):
                condition = layer(condition, t_emb)
            else:
                condition = layer(condition)
        # Return the condition embedding
        return condition


class ResFlow(Flow, L.LightningModule):
    """Class for the residual flow model."""

    def __init__(self, config: dict):
        super(ResFlow, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.nx = self.config.data.nx
        self.nc = self.config.data.nc
        self.nd = self.config.data.nd

        # flow config
        self.flow_config = self.config.flow
        self.sig_min = self.flow_config.sig_min
        self.time_emb_freq = self.flow_config.time_emb_freq
        self.latent_dim = self.flow_config.latent_dim
        self.num_centers = self.flow_config.num_centers

        # state embedding
        self.state_embedding = nn.Sequential(
            nn.Linear(self.nx + 2 * self.time_emb_freq, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, self.latent_dim),
        )

        # condition embedding
        self.condition_embedding = ConditionEmbedding(
            nc=self.nc,
            latent_dim=self.latent_dim,
            time_emb_freq=self.time_emb_freq,
        )

        # skip connections
        self.skip = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.latent_dim),
        )

        # domain embedding
        self.domain_embedding = RBFFiLM(
            num_centers=self.num_centers, latent_dim=self.latent_dim, nd=self.nd
        )

    def sample_base_density(self, x1: torch.Tensor, c: torch.Tensor):
        """sample from the base density"""
        return torch.randn_like(x1, device=x1.device)

    def sample_initial_condition(self, c: torch.Tensor, batch_size: int, n_gen: int):
        """get the initial condition for the flow"""
        return torch.randn(batch_size, n_gen, self.nx, device=self.device)

    def evaluate_vector_field(
        self, x: torch.Tensor, c: torch.Tensor, d: torch.Tensor, t: torch.Tensor
    ):
        """evalute the vector field of the flow"""
        # Embedd the time
        t_emb = self.time_embedding(t, self.time_emb_freq)
        # Embedd the state
        state_emb = self.state_embedding(torch.cat([x, t_emb], dim=-1))
        # Embedd the condition
        condition_emb = self.condition_embedding(c, t_emb)
        # Skip connection
        out = self.skip(state_emb + condition_emb) + state_emb + condition_emb
        # Embedd the domain
        out = self.domain_embedding(out, d)
        return out


if __name__ == "__main__":
    # get the data module
    data_module = get_data_module()
    # get the checkpointer
    ckp_save_path = config.checkpoint_save_path
    checkpointer = get_checkpointer(
        ckp_save_path + "/mfFlow" if config.mfFlow else ckp_save_path
    )
    # get the trainer
    logger_name = config.logger_name
    trainer = get_trainer(
        checkpointer=checkpointer,
        logger_name=logger_name + "_mfFlow" if config.mfFlow else logger_name,
        train_config=config.train,
    )
    # Model
    model = ResFlow(config)
    model.apply(init_weights)

    # Check if checkpoint is being loaded
    if config.checkpoint_load_path is not None:
        check_path(config.checkpoint_load_path)
        best_model_path = config.checkpoint_load_path
        printer(f"Loading checkpoint from {best_model_path}")
        # load the checkpoint
        model = ResFlow.load_from_checkpoint(best_model_path, map_location="cpu")
        model.to("cuda")

    # Train the model
    if config.train.stage == "train":
        printer("Starting training...")
        trainer.fit(model, data_module)
        # best model path
        best_model_path = checkpointer.best_model_path
        printer(f"Best model saved at {best_model_path}")
    elif config.train.stage == "eval":
        printer("Starting evaluation...")
        assert config.dataloader.reload is True, "Reload must be True for evaluation"
        check_path(config.checkpoint_load_path)
        best_model_path = config.checkpoint_load_path

    # Inference
    best_model = ResFlow.load_from_checkpoint(best_model_path, map_location="cpu")
    best_model.to("cuda" if torch.cuda.is_available() else "cpu")
    infer = Inference(
        model=best_model,
        test_config=data_module.test_config,
        statistics=data_module.statistics,
        job_name=config.job_name,
        mfFlow=config.mfFlow,
        generate_config=config.generate,
    )
    infer()
