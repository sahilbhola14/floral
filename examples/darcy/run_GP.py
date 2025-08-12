# examples/Darcy/run_FLOREN.py
"""
Gaussian Process Regression for Two-dimensional problem with non-linear correlation.
"""
import torch
import argparse
import gpytorch
from tqdm import tqdm
from omegaconf import OmegaConf
from mfFlow.utils import (
    printer,
    GPDataModule,
    InferenceGP,
)

from mfFlow.GP import build_gp

parser = argparse.ArgumentParser(description="Run Darcy with Gaussian Processes")
parser.add_argument(
    "--config",
    type=str,
    default="config_GP.yml",
    help="Path to the configuration file.",
)
args = parser.parse_args()
config = OmegaConf.load(args.config)
printer(f"Running Darcy with configuration: {args.config}")
printer(f"Using mfFlow: {config.mfFlow} and job name: {config.job_name}")
printer(
    f"Number of samples: {config.data.high_fidelity.n_samples} "
    f"sensors: {config.data.high_fidelity.n_sensors}"
)

torch.set_float32_matmul_precision("medium")  # for tensor cores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_module():
    """Get the data module for the Darcy problem."""
    data_module = GPDataModule(
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


def train_GP(model):
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    pbar = tqdm(
        range(config.train.max_epochs),
        desc="Training GP Model",
        unit="epoch",
        ncols=100,
    )
    for epoch in pbar:
        # Train step
        train_loss = model.train_step(mll, optimizer)
        # Valication step
        val_loss = model.val_step(mll)
        pbar.set_postfix({"train_loss": train_loss, "val_loss": val_loss})


if __name__ == "__main__":
    data_module = get_data_module()
    # gp regression model
    model = build_gp(
        train_set=data_module.train_set,
        val_set=data_module.val_set,
        device=device,
        gp_type="mini_batch_vanilla",
    ).to(device)
    # train the model
    train_GP(model)
    # infer the model
    infer = InferenceGP(
        model=model,
        test_config=data_module.test_config,
        statistics=data_module.statistics,
        job_name=config.job_name,
        mfFlow=config.mfFlow,
        device=device,
    )
    infer()
