# examples/poisson/run_FLOREN.py
"""
Flow-Matching Residual Embedded Nerual Opeartor for Poisson equation
"""
import torch
import torch.nn as nn
import argparse
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
from mfFlow.archs import RBFFiLM

parser = argparse.ArgumentParser(description="Run Poisson with specified parameters.")
parser.add_argument(
    "--config",
    type=str,
    default="config_FLOREN.yml",
    help="Path to the configuration file.",
)
args = parser.parse_args()
config = OmegaConf.load(args.config)
printer(f"Running Poisson with configuration: {args.config}")
printer(f"Using mfFlow: {config.mfFlow} and job name: {config.job_name}")

torch.set_float32_matmul_precision("medium")  # for tensor cores


def get_data_module():
    """Get the data module for the Poisson problem."""
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
            nn.Linear(self.nx + 2 * self.time_emb_freq, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.latent_dim),
        )

        # condition embedding
        self.condition_embedding = nn.Sequential(
            nn.Linear(self.nc + 2 * self.time_emb_freq, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.latent_dim),
        )

        # skip connections
        self.skip = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.latent_dim),
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
        condition_emb = self.condition_embedding(torch.cat([c, t_emb], dim=-1))
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
