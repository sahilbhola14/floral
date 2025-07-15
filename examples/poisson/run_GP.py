# examples/poisson/run_GP.py
"""
Gaussian Process Regression for Poisson equation
"""
import numpy as np
import torch
import argparse
import gpytorch
from tqdm import tqdm
from omegaconf import OmegaConf
from mfFlow.utils import (
    n2t,
    printer,
    GPDataModule,
    InferenceGP,
)

from mfFlow.GP import GPRegressionModel

parser = argparse.ArgumentParser(description="Run Poisson with Gaussian Processes")
parser.add_argument(
    "--config", type=str, default="config.yml", help="Path to the configuration file."
)
args = parser.parse_args()
config = OmegaConf.load(args.config)
printer(f"Running Poisson with configuration: {args.config}")

torch.set_float32_matmul_precision("medium")  # for tensor cores


def get_data_module():
    """Get the data module for the Poisson problem."""
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
        # Zero the gradients
        optimizer.zero_grad()
        # Train step
        train_loss = model.train_step(mll)
        # Backpropagation
        train_loss.backward()
        # Update the parameters
        optimizer.step()
        # Valication step
        val_loss = model.val_step(mll)
        pbar.set_postfix({"train_loss": train_loss.item(), "val_loss": val_loss.item()})


@torch.no_grad()
def infer_GP(model, likelihood, statistics):
    # Get the conditions
    test_data = np.load(config.data.test_data_path, allow_pickle=True)
    condition = n2t(test_data.get("condition", None))
    LF_field = n2t(test_data.get("LF_field", None))
    HF_field = n2t(test_data.get("HF_field", None))
    B, Nc = condition.shape
    domain = (
        n2t(test_data.get("domain", None)).unsqueeze(0).repeat(condition.shape[0], 1, 1)
    )
    in_features = torch.cat([condition.view(-1, 1), domain.view(-1, 1)], dim=-1).float()
    # Normalize the input features
    in_features = (in_features - statistics["in_features_train_mean"]) / statistics[
        "in_features_train_std"
    ]
    # Set to eval mode
    model.eval()
    likelihood.eval()
    # Infer
    with gpytorch.settings.fast_pred_var():
        pred = likelihood(model(in_features))
        pred_mean = pred.mean
        pred_std = pred.stddev
    # Denormalize the output
    pred_mean = (
        pred_mean.unsqueeze(-1) * statistics["out_features_train_std"]
        + statistics["out_features_train_mean"]
    )
    pred_std = pred_std.unsqueeze(-1) * statistics["out_features_train_std"]
    # Reshape
    pred_mean = pred_mean.view(B, Nc)
    pred_std = pred_std.view(B, Nc)
    # Results
    results = {
        "LF_field": LF_field,
        "HF_field": HF_field,
        "Prediction": {"mean": pred_mean, "std": pred_std},
    }
    # Save results
    torch.save(results, "test.pt")


if __name__ == "__main__":
    data_module = get_data_module()
    # gp regression model
    model = GPRegressionModel(
        train_set=data_module.train_set,
        val_set=data_module.val_set,
    )
    # train the model
    train_GP(model)
    # infer the model
    infer = InferenceGP(
        model=model,
        test_config=data_module.test_config,
        statistics=data_module.statistics,
        job_name=config.job_name,
        mfFlow=config.mfFlow,
    )
    infer()
