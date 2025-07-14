import numpy as np
import torch

# import torch.nn as nn
import argparse
import gpytorch

# import pytorch_lightning as L
# from tqdm import tqdm
from omegaconf import OmegaConf
from mfFlow.utils import (
    n2t,
    printer,
    # init_weights,
    # get_path,
    GPDataModule,
    # RunningAverageMeter,
)

# from mfFlow.GP import GPRegressionModel
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood

parser = argparse.ArgumentParser(description="Run oneDCorr with Gaussian Processes")
parser.add_argument(
    "--config", type=str, default="config.yml", help="Path to the configuration file."
)
args = parser.parse_args()
config = OmegaConf.load(args.config)
printer(f"Running oneDCorr with configuration: {args.config}")

torch.set_float32_matmul_precision("medium")  # for tensor cores


def get_data_module():
    """Get the data module for the oneDCorr problem."""
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


def get_datasets():
    path = config.data.high_fidelity.path
    data = np.load(path, allow_pickle=True)
    n_samples = config.data.high_fidelity.n_samples

    field = n2t(data.get("field")[:n_samples])
    condition = n2t(data.get("condition")[:n_samples])

    domain = n2t(data.get("domain"))
    domain_batch = domain.unsqueeze(0).repeat(n_samples, 1, 1)
    # In features
    in_features = torch.cat(
        [condition.view(-1, 1), domain_batch.view(-1, 1)], dim=-1
    ).float()
    # Out features
    out_features = field.view(-1, 1).float()

    # Split
    n_train = int(len(in_features) * config.dataloader.train_ratio)
    in_features_train, in_features_val = torch.split(
        in_features, [n_train, len(in_features) - n_train]
    )
    out_features_train, out_features_val = torch.split(
        out_features, [n_train, len(out_features) - n_train]
    )

    # Normalize
    in_features_train_mean = in_features_train.mean(dim=0, keepdim=True)
    in_features_train_std = in_features_train.std(dim=0, keepdim=True)
    out_features_train_mean = out_features_train.mean(dim=0, keepdim=True)
    out_features_train_std = out_features_train.std(dim=0, keepdim=True)
    in_features_train = (
        in_features_train - in_features_train_mean
    ) / in_features_train_std
    out_features_train = (
        out_features_train - out_features_train_mean
    ) / out_features_train_std
    in_features_val = (in_features_val - in_features_train_mean) / in_features_train_std
    out_features_val = (
        out_features_val - out_features_train_mean
    ) / out_features_train_std

    # Statistics
    statistics = {
        "in_features_train_mean": in_features_train_mean,
        "in_features_train_std": in_features_train_std,
        "out_features_train_mean": out_features_train_mean,
        "out_features_train_std": out_features_train_std,
    }

    # Create datasets
    train_set = torch.utils.data.TensorDataset(in_features_train, out_features_train)
    val_set = torch.utils.data.TensorDataset(in_features_val, out_features_val)

    return train_set, val_set, statistics


def train_GP(model, likelihood):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for ii in range(config.train.max_epochs):
        optimizer.zero_grad()
        output = model(model.train_x)
        loss = -mll(output, model.train_y)
        loss.backward()
        optimizer.step()
        print(f"Iter {ii + 1}/{config.train.max_epochs} - Loss: {loss.item(): .3f}")


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


class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        self.train_x = train_x
        self.train_y = train_y.squeeze()
        super(GPModel, self).__init__(self.train_x, self.train_y, likelihood)
        self.mean = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == "__main__":
    data_module = get_data_module()
    train_set, val_set, statistics = get_datasets()
    train_x, train_y = train_set.tensors
    likelihood = GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    train_GP(model, likelihood)
    infer_GP(model, likelihood, statistics)
