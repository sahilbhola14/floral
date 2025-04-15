import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as L

# import argparse
import utils.utils as utils

# import os.path as osp

import torch.nn as nn

from types import SimpleNamespace
from sklearn import gaussian_process as gp
from scipy import interpolate
from src.flow import Flow
from torch.utils.data import DataLoader, TensorDataset

# from omegaconf import OmegaConf
# from src.archs.encoding import RBFFiLM

torch.set_float32_matmul_precision("medium")  # for tensor cores

# Begin Parameters
N_SAMPLES = 500
LOW_CONFIG = {"M": 10}
HIGH_CONFIG = {"M": 100}
RESIDUAL = True
# End Parameters


def solver(f, N):
    """solver for the Poisson equation"""
    h = 1 / (N - 1)
    K = -2 * np.eye(N - 2) + np.eye(N - 2, k=1) + np.eye(N - 2, k=-1)
    b = h**2 * 20 * f[1:-1]
    u = np.linalg.solve(K, b)
    u = np.concatenate(([0], u, [0]))
    return u


# GP
class GRF:
    def __init__(self, T, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.T = T
        self.kernel = kernel
        self.length_scale = length_scale
        self.N = N
        self.x = np.linspace(0, T, self.N)[:, None]
        self.interp = interp
        kernel = gp.kernels.RBF(length_scale=self.length_scale)
        self.K = kernel(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n):
        """generate a random field sample"""
        return np.dot(self.L, np.random.randn(self.N, n)).T

    def eval_u(self, ys, sensors):
        """evaluate the random field at the sensor locations"""
        y_interp = []
        for y in ys:
            interp = interpolate.interp1d(
                self.x.ravel(), y, kind=self.interp, copy=False, assume_sorted=True
            )
            y_interp.append(interp(sensors))

        return np.array(y_interp)


# DataGeneration
def generate_data():
    space = GRF(1, length_scale=0.05, N=1000, interp="cubic")
    features = space.random(N_SAMPLES)
    domain_high = np.linspace(0, 1, HIGH_CONFIG.get("M"))
    domain_low = np.linspace(0, 1, LOW_CONFIG.get("M"))
    features_high = space.eval_u(features, domain_high)
    features_low = space.eval_u(features, domain_low)

    # High Fidelity Data
    x_high, y_high, y_high_domain = [], [], []
    for ii in range(N_SAMPLES):
        sol = solver(features_high[ii], HIGH_CONFIG.get("M"))
        idx = np.random.choice(HIGH_CONFIG.get("M"), 1, replace=False)
        x_high.append(domain_high[idx].item())
        y_high.append(sol[idx].item())
        y_high_domain.append(sol)

    x_high = np.array(x_high)  # High fidelity random sensor
    y_high = np.array(y_high)  # High fidelity solution at random sensor
    y_high_domain = np.array(y_high_domain)  # High fidelity solution

    high_data = {}
    high_data["x_high"] = x_high
    high_data["y_high"] = y_high
    high_data["y_high_at_domain"] = y_high_domain
    high_data["features"] = features_high
    high_data["domain"] = domain_high

    # Low Fidelity Data
    y_low, y_low_at_x_high, y_low_at_domain_high = [], [], []
    for ii in range(N_SAMPLES):
        sol = solver(features_low[ii], LOW_CONFIG.get("M"))
        interp = interpolate.interp1d(
            domain_low, sol, kind="cubic", copy=False, assume_sorted=True
        )
        y_low_at_x_high.append(interp(x_high[ii]))
        y_low_at_domain_high.append(interp(domain_high))
        y_low.append(sol)

    y_low_at_x_high = np.array(
        y_low_at_x_high
    )  # Low fidelity solution at high random sensor
    y_low_at_domain_high = np.array(
        y_low_at_domain_high
    )  # Low fidelity solution at high domain
    y_low = np.array(y_low)  # Low fideltiy solution

    low_data = {}
    low_data["y_low_at_x_high"] = y_low_at_x_high
    low_data["y_low_at_domain_high"] = y_low_at_domain_high
    low_data["y_low"] = y_low
    low_data["features"] = features_low
    low_data["domain"] = domain_low

    return low_data, high_data


def comp_qoi(u, dx):
    # Compute the gradient
    du_dx = np.gradient(u, dx)
    # Energy
    energy_density = 0.5 * (du_dx**2)
    return np.sum(energy_density) * dx


def compute_true_qoi(n_samples=500000):
    """
    Compute the true qoi q = E[P], where a realization of p is an integral QoI
    over the domain
    """
    space = GRF(1, length_scale=0.05, N=1000, interp="cubic")
    features = space.random(n_samples)
    domain_high = np.linspace(0, 1, HIGH_CONFIG.get("M"))
    dx = domain_high[1] - domain_high[0]
    features_high = space.eval_u(features, domain_high)
    p_samples = []
    # weight = np.exp(-10 * (domain_high - 0.5 )** 2)
    for ii in range(n_samples):
        # compute the solution
        sol = solver(features_high[ii], HIGH_CONFIG.get("M"))
        p_samples.append(comp_qoi(sol, dx))

    qoi = np.mean(np.array(p_samples))
    print(f"Integral QoI: {qoi} using {n_samples} Monte-Carlo samples")
    np.save("integral_qoi.npy", qoi)
    return qoi


# DataModule
class dataModule(L.LightningDataModule):
    def __init__(self, low_data, high_data, p_train=0.7):
        super().__init__()
        self.low_data = low_data
        self.high_data = high_data
        self.p_train = p_train

    def setup(self, stage=None):
        # Field
        high_field = utils.n2t(self.high_data.get("y_high")).view(-1, 1)
        high_field_at_domain = utils.n2t(self.high_data.get("y_high_at_domain")).view(
            -1, HIGH_CONFIG.get("M")
        )

        low_field = utils.n2t(low_data.get("y_low_at_x_high")).view(-1, 1)
        low_field_at_domain_high = utils.n2t(low_data.get("y_low_at_domain_high")).view(
            -1, HIGH_CONFIG.get("M")
        )

        domain = utils.n2t(high_data.get("x_high")).view(-1, 1)

        if RESIDUAL:
            field = high_field - low_field
        else:
            field = high_field
        # Condition
        condition = utils.n2t(self.high_data.get("features"))

        # Split for training and validation
        n_train = int(N_SAMPLES * self.p_train)
        field_train, field_val = field[:n_train], field[n_train:]
        condition_train, condition_val = condition[:n_train], condition[n_train:]
        domain_train, domain_val = domain[:n_train], domain[n_train:]

        # Normalize
        field_mean, field_std = field_train.mean(0), field_train.std(0)
        condition_mean, condition_std = condition_train.mean(0), condition_train.std(0)

        field_train = (field_train - field_mean) / field_std  # Normalize train field
        field_val = (field_val - field_mean) / field_std  # Normalize val field

        condition_train = (
            condition_train - condition_mean
        ) / condition_std  # Normalize train condition
        condition_val = (
            condition_val - condition_mean
        ) / condition_std  # Normalize val condition

        # Testing Config
        self.test_config = {}

        high_field_at_domain_val = high_field_at_domain[n_train:]

        low_field_at_domain_high_val = low_field_at_domain_high[n_train:]

        self.test_config["domain"] = utils.n2t(self.high_data.get("domain")).view(-1, 1)
        self.test_config["condition"] = condition_val
        self.test_config["high_field"] = high_field_at_domain_val
        self.test_config["low_field"] = low_field_at_domain_high_val
        self.test_config["field_stats"] = {"mean": field_mean, "std": field_std}
        self.test_config["condition_stats"] = {
            "mean": condition_mean,
            "std": condition_std,
        }

        # Dataset
        self.train_set = TensorDataset(field_train, condition_train, domain_train)
        self.val_set = TensorDataset(field_val, condition_val, domain_val)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=64, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=64, shuffle=False)


# ResFlow
class ResFlow(Flow, L.LightningModule):
    def __init__(self, nx: int, nc: int, nd: int):
        super(ResFlow, self).__init__()
        self.save_hyperparameters()
        self.nx = nx
        self.nc = nc
        self.nd = nd

        self.sig_min = 1e-5
        self.n_freq = 4
        self.latent_dim = 32

        # State embedding
        self.state_encoder = nn.Sequential(
            nn.Linear(self.nx + 2 * self.n_freq, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, self.latent_dim),
        )

        # Conditional embedding
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.nc + 2 * self.n_freq, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, self.latent_dim),
        )

        # Skip connection
        self.skip = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.latent_dim),
        )

        # Domain Encoder
        num_centers = 10
        self.centers = nn.Parameter(torch.linspace(0, 1, num_centers))
        self.domain_encoder = nn.Sequential(
            nn.Linear(num_centers, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.latent_dim),
        )

        self.bias = nn.Parameter(torch.zeros(self.nx))

    def rbf_encoding(self, mod, gamma=10.0):
        return torch.exp(-gamma * (mod - self.centers) ** 2)

    def sample_base_density(self, x1: torch.Tensor, c: torch.Tensor):
        """sample the base density"""
        return torch.randn_like(x1, device=x1.device)

    def evaluate_vector_field(
        self, x: torch.Tensor, c: torch.Tensor, d: torch.Tensor, t: torch.Tensor
    ):
        """evaluate the vector field"""
        # Encode the time
        enc_time = self.time_embedding(t, self.n_freq)
        # Encode the state
        enc_state = self.state_encoder(torch.cat((x, enc_time), dim=-1))
        # Encode the condition
        enc_condition = self.condition_encoder(torch.cat((c, enc_time), dim=-1))
        # Skip
        enc_skip = self.skip(enc_state + enc_condition)
        out = enc_skip + enc_state + enc_condition
        # Encode the domain
        rbf_encoding = self.rbf_encoding(d)
        enc_domain = self.domain_encoder(rbf_encoding)

        return torch.sum(out * enc_domain, dim=-1, keepdims=True) + self.bias

    def sample_initial_condition(self, c: torch.Tensor, batch_size: int, n_gen: int):
        """get the initial condition for the flow"""
        return torch.randn(batch_size, n_gen, self.nx, device=self.device)

    def append_boundary_conditions(self, x: torch.Tensor):
        """append the boundary conditions to the data"""
        pass

    def remove_boundary_conditions(self, x: torch.Tensor):
        """remove the boundary conditions to the data"""
        pass


if __name__ == "__main__":
    # Compute the true QoI
    compute_true_qoi()
    # generate data
    low_data, high_data = generate_data()
    # DataModule
    data_module = dataModule(low_data, high_data)
    # Model
    model = ResFlow(nx=1, nc=HIGH_CONFIG.get("M"), nd=1)
    # checkpointer
    checkpointer = utils.get_checkpointer("./experiments/mfFlow/Poisson/checkpoints")
    # Trainer
    train_config = SimpleNamespace(
        **{"max_epochs": 10000, "devices": 2, "accelerator": "gpu", "strategy": "ddp"}
    )
    trainer = utils.get_trainer(
        checkpointer=checkpointer, logger_name="mfFlow", train_config=train_config
    )
    # Train
    trainer.fit(model, data_module)

    # Load best model
    best_model_path = checkpointer.best_model_path
    model = ResFlow.load_from_checkpoint(best_model_path)

    # Test
    fig, axs = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    axs = axs.ravel()
    d_eval = data_module.test_config.get("domain")
    for ii in range(len(axs) // 2):
        c_eval = data_module.test_config.get("condition")[ii].view(1, -1)
        x1_true = data_module.test_config.get("high_field")[ii]
        low_field = data_module.test_config.get("low_field")[ii]
        x1_hat = model.interpolate(c_eval, d_eval).squeeze(-1).T.detach().to("cpu")
        if RESIDUAL:
            # Denormalize
            pred_residual = (
                x1_hat * data_module.test_config["field_stats"]["std"]
            ) + data_module.test_config["field_stats"]["mean"]
            x1_hat = pred_residual + low_field

        # True Residual
        true_residual = x1_true - low_field

        mean_pred = utils.t2n(x1_hat.mean(0))
        std_pred = utils.t2n(x1_hat.std(0))

        # Field Plot
        axs[ii].plot(utils.t2n(d_eval), utils.t2n(x1_true), label="True", color="blue")
        axs[ii].plot(utils.t2n(d_eval), utils.t2n(low_field), label="Low", color="red")
        axs[ii].plot(utils.t2n(d_eval), mean_pred, label="CorrFlow", color="green")
        axs[ii].fill_between(
            utils.t2n(d_eval).ravel(),
            mean_pred - std_pred,
            mean_pred + std_pred,
            alpha=0.2,
            color="green",
        )
        axs[ii].set_xlabel(r"x")
        axs[ii].set_ylabel(r"u(x,f(x))")
        axs[ii].label_outer()
        if ii == 0:
            axs[ii].legend()

        # Residual
        axs[ii + 5].plot(
            utils.t2n(d_eval), utils.t2n(true_residual), label="True", color="blue"
        )
        axs[ii + 5].plot(
            utils.t2n(d_eval),
            utils.t2n(pred_residual.mean(0)),
            label="CorrFlow",
            color="green",
        )
        axs[ii + 5].fill_between(
            utils.t2n(d_eval).ravel(),
            pred_residual.mean(0) - pred_residual.std(0),
            pred_residual.mean(0) + pred_residual.std(0),
            alpha=0.2,
            color="green",
        )
        axs[ii + 5].set_xlabel(r"x")
        axs[ii + 5].set_ylabel(r"residual")
        axs[ii + 5].label_outer()

    plt.tight_layout()
    plt.savefig("corrFlow.png")
