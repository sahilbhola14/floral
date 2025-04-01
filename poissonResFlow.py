import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as L
import argparse
import utils.utils as utils
import os.path as osp

import torch.nn as nn

from scipy.sparse import diags

from scipy.sparse.linalg import spsolve
from src.flow import Flow
from torch.utils.data import DataLoader, Dataset, TensorDataset
from omegaconf import OmegaConf
from src.archs.encoding import RBFFiLM

parser = argparse.ArgumentParser(
    description="Multi-Fidelity Flow Matching for Poisson Equation"
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/config_poisson.yml",
    help="Path to the config file",
)
args = parser.parse_args()
config = OmegaConf.load(args.config)
utils.printer(f"Running with config: {args.config}")

torch.set_float32_matmul_precision("medium")  # for tensor cores


class Poisson:
    """
    Solve the differential equation: d²u/dx² + f(x;θ) = 0
    with boundary conditions u(0) = 0, u(1) = 0,
    where f(x;θ) = sum_{i=1}^n θ_i sin(2πi x)
    """

    def __init__(self, config, modes=5):
        self.n_pts = config.get("n_pts")
        self.n_samples = config.get("n_samples")
        self.n_modes = modes  # Number of modes for the forcing term
        self.L = 1.0  # domain length
        self.boundary_conditions = {
            "domain": [0.0, 1.0],
            "value": [0.0, 0.0],
        }  # Values of boundary conditions
        self.domain_full = torch.linspace(0, self.L, self.n_pts)  # full domain
        self.domain_interior = self.domain_full[1:-1]  # interior domain
        self.nx = self.n_pts - 2  # Number of points in the interior domain
        self.nc = self.n_modes  # Number of modes for the forcing term
        self.nd = 1  # Dimension of the domain

    def compute_dataset(self):
        """Function computes the dataset"""
        self.condition = self.sample_theta()  # condition for the field
        self.field = torch.zeros(self.n_samples, self.nx)  # field

        for ii in range(self.n_samples):
            u = self.solve(self.condition[ii])
            self.field[ii] = utils.n2t(u)[1:-1]  # Only interior field is generated

    def sample_theta(self):
        """Sample the model parameters"""
        return torch.randn(self.n_samples, self.n_modes)

    def get_rhs(self, x, theta):
        """function computes the rhs terms"""
        f_x = torch.zeros_like(x)
        for n in range(1, self.n_modes + 1):
            f_x += theta[n - 1] * torch.sin(2.0 * torch.pi * n * x)
        return f_x

    def solve(self, theta):
        N = self.n_pts - 2  # Number of interior points
        x = self.domain_interior  # interior points
        dx = self.L / (N + 1)
        A = (
            diags([1, -2, 1], [-1, 0, 1], shape=(N, N)) / dx**2
        )  # Discretized Laplacian
        f = self.get_rhs(x, theta)  # Forcing term
        u = spsolve(A, -f)  # Solve linear system
        return np.concatenate(([0], u, [0]))  # Apply boundary conditions

    def get_true_solution(self, theta, n_pts):
        """function computes the interpolation config"""
        N = n_pts - 2  # Number of interior points

        domain_full = torch.linspace(0, self.L, n_pts)  # full domain
        domain_interior = domain_full[1:-1]  # interior domain

        x = domain_interior  # interior points
        dx = self.L / (N + 1)
        A = (
            diags([1, -2, 1], [-1, 0, 1], shape=(N, N)) / dx**2
        )  # Discretized Laplacian
        f = self.get_rhs(x, theta)  # Forcing term
        u = spsolve(A, -f)  # Solve linear system

        return np.concatenate(([0], u, [0])), domain_full, domain_interior


class CustomDataset(Dataset):
    def __init__(self, field, condition, domain):
        self.field = field
        self.condition = condition
        self.domain = domain
        self.tensors = (self.field, self.condition, self.domain)

    def __len__(self):
        return len(self.field)

    def __getitem__(self, idx):
        return self.field[idx], self.condition[idx], self.domain


class sourceFlow(Flow, L.LightningModule):
    """SOURCE Flow Model"""

    def __init__(
        self,
        config: dict,
        nx: int,
        nc: int,
        nd: int,
        domain: dict,
        normalization_config_file: str,
        boundary_conditions: dict,
    ):
        super(sourceFlow, self).__init__()
        self.save_hyperparameters()

        self.config = config  # config
        self.nx = nx  # dimension of the field
        self.nc = nc  # dimension of the conditional information
        self.nd = nd  # dimension of the domain
        self.domain_full = torch.FloatTensor(domain.get("full")).view(-1, self.nd)
        self.domain_interior = torch.FloatTensor(domain.get("interior")).view(
            -1, self.nd
        )
        self.flow_config = self.config.flow
        self.sig_min = self.flow_config.sig_min
        self.latent_dim = self.flow_config.latent_dim
        self.n_freq = self.flow_config.time_emb_freq
        self.normalization_config = torch.load(normalization_config_file)
        self.boundary_conditions = boundary_conditions

        # State embedding
        self.state_encoder = nn.Sequential(
            nn.Linear(self.nx + 2 * self.n_freq, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
        )

        # Conditional embedding
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.nc + 2 * self.n_freq, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
        )

        # Skip connection
        self.skip = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
        )

        # Domain encoder
        self.domain_encoder = RBFFiLM(in_dim=self.nd, out_dim=self.latent_dim)

    def sample_base_density(self, x1: torch.Tensor, c: torch.Tensor):
        """sample the base density"""
        x0 = torch.randn_like(x1, device=self.device)
        return x0

    def evaluate_vector_field(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        d_eval: torch.Tensor = None,
    ):
        """evaluate the vector field"""
        # encode time
        enc_time = self.time_embedding(t, self.n_freq)
        # encode the state
        enc_state = self.state_encoder(torch.cat([x, enc_time], dim=-1))
        # encode the condition
        enc_cond = self.condition_encoder(torch.cat([c, enc_time], dim=-1))
        # skip connection
        skip = self.skip(enc_state + enc_cond)
        out = enc_state + enc_cond + skip
        # Encode the domain
        if d_eval is None:
            # For training, always the interior domain is used
            out = self.domain_encoder(out, self.domain_interior.to(self.device))
        else:
            # Model can be queried at any domain
            out = self.domain_encoder(out, d_eval.to(self.device))
        return out

    def sample_initial_condition(self, c: torch.Tensor, batch_size: int, n_gen: int):
        """sample the initial condition"""
        x0 = torch.randn(batch_size, n_gen, self.nx, device=self.device)
        return x0

    def append_boundary_conditions(self, x: torch.Tensor):
        """function appends the boundary conditon"""
        left_boundary = self.boundary_conditions.get("value")[0] * torch.ones(
            x.shape[0], 1, device=self.device
        )
        right_boundary = self.boundary_conditions.get("value")[1] * torch.ones(
            x.shape[0], 1, device=self.device
        )
        x = torch.cat([left_boundary, x, right_boundary], dim=-1)
        return x

    def remove_boundary_conditions(self, x: torch.Tensor):
        """remove the boundary conditions"""
        return x[:, 1:-1]


class residualFlow(Flow, L.LightningModule):
    """RESIDUAL Flow Model"""

    def __init__(
        self,
        config: dict,  # configuration
        nx: int,  # dimension of the field (interior)
        nc: int,  # dimension of the conditions
        nd: int,  # dimension of the domain
        domain: dict,  # domain information (interior and full)
        normalization_config_file: str,  # normalization config file
        boundary_conditions: dict,  # boundary conditions
        best_source_model_path: str,  # path to the best source model
    ):
        super(residualFlow, self).__init__()
        self.save_hyperparameters()

        self.config = config  # config
        self.nx = nx  # dimension of the field
        self.nc = nc  # dimension of the conditional information
        self.nd = nd  # dimension of the domain
        self.domain_full = torch.FloatTensor(domain.get("full")).view(-1, self.nd)
        self.domain_interior = torch.FloatTensor(domain.get("interior")).view(
            -1, self.nd
        )
        self.flow_config = self.config.flow
        self.sig_min = self.flow_config.sig_min
        self.latent_dim = self.flow_config.latent_dim
        self.n_freq = self.flow_config.time_emb_freq
        self.normalization_config = torch.load(normalization_config_file)
        self.boundary_conditions = boundary_conditions

        # source flow
        self.best_source_model_path = best_source_model_path
        self.MF2M = self.config.data.MF2M
        self.source_model = self.initialize_source_flow_model()

        # State embedding
        self.state_encoder = nn.Sequential(
            nn.Linear(self.nx + 2 * self.n_freq, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.nx),
        )

        # Conditional embedding
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.nc + 2 * self.n_freq, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.nx),
        )

        # Skip connection
        self.skip = nn.Sequential(
            nn.Linear(self.nx, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, self.nx),
        )

        # Domain encoder
        # self.domain_encoder = RBFFiLM(in_dim=self.nd, out_dim=self.latent_dim)

    def initialize_source_flow_model(self):
        """Initialize the source flow model (set to eval mode)"""
        if self.best_source_model_path is not None:
            source_model = sourceFlow.load_from_checkpoint(self.best_source_model_path)
            assert osp.exists(self.best_source_model_path), "Invalid source path"
            for param in source_model.parameters():
                param.requires_grad = False
            return source_model
        else:
            return None

    def restrict_sample(self, x: torch.Tensor):
        """Restrict the sample to the source model domain using Grid Interpolation
        Args:
            x (torch.Tensor): normalized sample tensor
        Returns:
            x_restrict (torch.Tensor): restricted sample tensor (denormalized)
        """
        # Evalute the sample at the source flow domain
        target_domain = self.source_model.domain_full
        # Denormalize the sample
        x_denormal, _ = self.denormalize_data(x=x)
        # Append the boundary conditions
        x_full = self.append_boundary_conditions(x_denormal)
        # Restrict the sample to the source model domain
        x_restrict = utils.restrict_domain(x_full, self.domain_full, target_domain)
        # Remove the boundary conditions
        x_restrict = self.source_model.remove_boundary_conditions(x_restrict)

        assert x_restrict.shape == (
            x.shape[0],
            self.source_model.nx,
        ), "Invalid dimensions"

        return x_restrict

    @torch.no_grad()
    def evaluate_source_vector_field(
        self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor
    ):
        """Function evalutes the vector field using the source model"""
        # Denormalize and Restrict the sample to the source model domain
        x_denormal = self.restrict_sample(x)

        # Denormalize the condition (to be used in the source model)
        _, c_denormal = self.denormalize_data(c=c)

        # Normalize the restricted sample according to the source model
        x_normal, c_normal = self.source_model.normalize_data(
            x=x_denormal, c=c_denormal
        )
        # Evalute the source model vector field
        vt_source = self.source_model.evaluate_vector_field(
            x_normal, c_normal, t, d_eval=self.domain_interior
        )

        return vt_source

    @torch.no_grad()
    def sample_base_density(self, x1: torch.Tensor, c: torch.Tensor):
        """sample the base density"""
        if self.best_source_model_path is None:
            x0 = torch.randn_like(x1, device=self.device)
        else:
            _, c_denormal = self.denormalize_data(
                c=c
            )  # low fideltiy query takes in denormalized condition
            interpolation_config = {
                "interpolate": True,
                "target_domain": self.domain_full,
            }
            # Generate one sample from the (trained) source model
            x0_denormal = self.source_model.query(
                c_denormal, interpolation_config=interpolation_config, n_gen=1
            ).squeeze(1)

            assert x0_denormal.shape == x1.shape, "Invalid dimensions"

            # Normalize according to the residual model
            x0, _ = self.normalize_data(x=x0_denormal)

            # Add noise
            x0 += 1e-1 * torch.randn_like(x0, device=self.device)

        return x0

    def evaluate_vector_field(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        """evaluate the vector field"""

        if self.MF2M:
            # Define the vector field using the source model vector field
            vt_source = self.evaluate_source_vector_field(x, c, t)

            # Residual vector field

            # encode time
            enc_time = self.time_embedding(t, self.n_freq)
            # encode the state
            enc_state = self.state_encoder(torch.cat([vt_source, enc_time], dim=-1))
            # encode the condition
            enc_cond = self.condition_encoder(torch.cat([c, enc_time], dim=-1))
            # skip connection
            skip = self.skip(enc_state + enc_cond)

            out = vt_source + enc_cond + skip

        else:
            # encode time
            enc_time = self.time_embedding(t, self.n_freq)
            # encode the state
            enc_state = self.state_encoder(torch.cat([x, enc_time], dim=-1))
            # encode the condition
            enc_cond = self.condition_encoder(torch.cat([c, enc_time], dim=-1))
            # skip connection
            skip = self.skip(enc_state + enc_cond)

            out = enc_state + enc_cond + skip

        return out

    def sample_initial_condition(self, c: torch.Tensor, batch_size: int, n_gen: int):
        """sample the initial condition"""
        if self.best_source_model_path is None:
            x0 = torch.randn(batch_size, n_gen, self.nx, device=self.device)
        else:
            _, c_denormal = self.denormalize_data(
                c=c
            )  # low fideltiy query takes in denormalized condition
            interpolation_config = {
                "interpolate": True,
                "target_domain": self.domain_full,
            }
            # Generate one sample from the (trained) source model
            x0_denormal = self.source_model.query(
                c_denormal, interpolation_config=interpolation_config, n_gen=n_gen
            ).squeeze(1)

            # Normalize according to the residual model
            x0, _ = self.normalize_data(x=x0_denormal)

            # Add noise
            x0 += 1e-1 * torch.randn_like(x0, device=self.device)

            assert x0.shape == (batch_size, n_gen, self.nx), "Invalid dimensions"

        return x0

    def append_boundary_conditions(self, x: torch.Tensor):
        """function appends the boundary conditon"""
        left_boundary = self.boundary_conditions.get("value")[0] * torch.ones(
            x.shape[0], 1, device=self.device
        )
        right_boundary = self.boundary_conditions.get("value")[1] * torch.ones(
            x.shape[0], 1, device=self.device
        )
        x = torch.cat([left_boundary, x, right_boundary], dim=-1)
        return x

    def remove_boundary_conditions(self, x: torch.Tensor):
        """remove the boundary conditions"""
        return x[:, 1:-1]


class dataModule(L.LightningDataModule):
    def __init__(self, config, model="low_fidelity"):
        super(dataModule, self).__init__()
        self.config = config
        self.model = model
        self.data_config = (
            self.config.data.low_fidelity
            if model == "low_fidelity"
            else self.config.data.high_fidelity
        )
        self.n_samples = self.data_config.n_samples  # number of samples (train + val)
        self.loader_config = (
            self.config.dataloader.low_fidelity
            if model == "low_fidelity"
            else self.config.dataloader.high_fidelity
        )
        self.poisson = Poisson(
            self.data_config, modes=self.config.data.modes
        )  # Solve the equations

        self.nx = self.poisson.nx
        self.nc = self.poisson.nc
        self.nd = self.poisson.nd
        self.domain = {
            "full": self.poisson.domain_full.tolist(),
            "interior": self.poisson.domain_interior.tolist(),
        }

        self.boundary_conditions = self.poisson.boundary_conditions

        self.normalization_config = {}

        self.dataset_file = "dataset_" + self.model + ".pt"
        self.normalization_file = "normalization_config_" + self.model + ".pt"
        self.setup()

    def normalize_dataset(self, field, condition):
        """Normalize the dataset"""
        condition_stats = self.normalization_config["condition"]
        field_stats = self.normalization_config["field"]

        condition_mean, condition_std = condition_stats.get(
            "mean"
        ), condition_stats.get("std")
        field_mean, field_std = field_stats.get("mean"), field_stats.get("std")

        assert (
            condition.shape[-1] == len(condition_mean) == len(condition_std)
        ), "Invalid dimensions"
        assert (
            field.shape[-1] == len(field_mean) == len(field_std)
        ), "Invalid dimensions"

        # Normalize the condition
        condition = (condition - condition_mean) / condition_std
        # Normalize the field
        field = (field - field_mean) / field_std

        return field, condition

    def comp_dataset_stats(self, field_train, condition_train):
        """function computes the dataset statistics"""
        condition_mean = condition_train.mean(0)
        condition_std = condition_train.std(0)

        field_mean = field_train.mean(0)
        field_std = field_train.std(0)

        field_stats = {}
        field_stats["mean"] = field_mean
        field_stats["std"] = field_std

        condition_stats = {}
        condition_stats["mean"] = condition_mean
        condition_stats["std"] = condition_std

        self.normalization_config["field"] = field_stats
        self.normalization_config["condition"] = condition_stats

    def get_interpolation_config(self, condition_val):
        """get interpolation config"""
        # True/High-fideltiy solution
        n_pts_high = self.config.data.high_fidelity.n_pts
        field_true = torch.zeros(len(condition_val), n_pts_high - 2)
        for ii in range(len(condition_val)):
            (
                u_true,
                domain_full_true,
                domain_interior_true,
            ) = self.poisson.get_true_solution(condition_val[ii], n_pts=n_pts_high)
            field_true[ii] = utils.n2t(u_true)[1:-1]  # removed the boundary

        # Low-fideltiy solution
        n_pts_low = self.config.data.low_fidelity.n_pts
        field_coarse = torch.zeros(len(condition_val), n_pts_low - 2)
        for ii in range(len(condition_val)):
            (
                u_true,
                domain_full_coarse,
                domain_interior_coarse,
            ) = self.poisson.get_true_solution(condition_val[ii], n_pts=n_pts_low)
            field_coarse[ii] = utils.n2t(u_true)[1:-1]  # removed the boundary

        interpolation_config = {}
        interpolation_config["field_true"] = field_true
        interpolation_config["domain_full_true"] = domain_full_true.view(-1, 1)
        interpolation_config["domain_interior_true"] = domain_interior_true.view(-1, 1)

        interpolation_config["field_coarse"] = field_coarse
        interpolation_config["domain_full_coarse"] = domain_full_coarse.view(-1, 1)
        interpolation_config["domain_interior_coarse"] = domain_interior_coarse.view(
            -1, 1
        )

        interpolation_config["condition"] = condition_val

        return interpolation_config

    def setup(self, stage=None):
        """setup the data"""
        if self.data_config.load_dataset is False:
            # Solve the poisson equation
            self.poisson.compute_dataset()

            # Extract the data
            field = self.poisson.field
            condition = self.poisson.condition

            # split
            n_train = int(self.n_samples * self.loader_config.train_ratio)
            condition_train, field_train = condition[:n_train], field[:n_train]
            condition_val, field_val = condition[n_train:], field[n_train:]

            # get the interpolation config (must use unnormalized condition)
            self.interpolation_config = self.get_interpolation_config(condition_val[:4])

            # Compute the statstistics
            self.comp_dataset_stats(field_train, condition_train)

            # Normalize the data
            field_train, condition_train = self.normalize_dataset(
                field_train, condition_train
            )
            field_val, condition_val = self.normalize_dataset(field_val, condition_val)

            # Create the datasets
            self.train_set = TensorDataset(field_train, condition_train)
            self.val_set = TensorDataset(field_val, condition_val)

            # Save
            save_data_dict = {
                "train": self.train_set,
                "val": self.val_set,
                "interpolation_config": self.interpolation_config,
            }

            torch.save(save_data_dict, self.dataset_file)
            torch.save(self.normalization_config, self.normalization_file)

        else:
            data_dict = torch.load(self.dataset_file)
            self.train_set = data_dict.get("train")
            self.val_set = data_dict.get("val")
            self.interpolation_config = data_dict.get("interpolation_config")

            self.normalization_config = torch.load(self.normalization_file)

    def collate_fn(self, batch):
        """Function for pre processing the batch"""
        field, condition, domain = zip(*batch)
        field = torch.stack(field)
        condition = torch.stack(condition)
        domain = domain[0]
        return field, condition, domain

    def train_dataloader(self):
        """train dataloader"""
        return DataLoader(
            self.train_set,
            batch_size=self.loader_config.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        """val dataloader"""
        return DataLoader(
            self.val_set,
            batch_size=self.loader_config.batch_size,
            shuffle=False,
        )


def train_source_model():
    """train the source(low) fidelity model"""
    # Specs
    model = "low_fidelity"
    data_config = config.data.low_fidelity
    train_config = config.train.low_fidelity

    data_module = dataModule(config, model=model)
    model = sourceFlow(
        config,
        nx=data_module.nx,
        nc=data_module.nc,
        nd=data_module.nd,
        domain=data_module.domain,
        normalization_config_file=data_module.normalization_file,
        boundary_conditions=data_module.boundary_conditions,
    )
    checkpointer = utils.get_checkpointer(data_config.checkpoint_path)
    trainer = utils.get_trainer(
        checkpointer=checkpointer,
        logger_name=data_config.logger_name,
        train_config=train_config,
    )

    if train_config.mode == "train":
        # Train
        trainer.fit(model, data_module)
        best_model_path = checkpointer.best_model_path
    elif train_config.mode == "eval":
        # Load the checkpoint
        assert (
            data_config.load_checkpoint is not None
        ), "For eval mode, a checkpoint must be provided"
        assert (
            data_config.load_dataset is True
        ), "For using past checkpoint, the relevant dataset must be loaded"
        best_model_path = (
            data_config.checkpoint_path + "/" + data_config.load_checkpoint
        )
        assert osp.exists(best_model_path), "Config file unavailable (Typo?)"
    else:
        raise ValueError("Invalid mode")

    # Load the best model
    model = sourceFlow.load_from_checkpoint(best_model_path)
    # Evaluate the validation dataset
    # model.evaluate_dataset(
    #     data_module.val_set, plot=True, n_gen=100
    # )  # get the prediction

    # Query the model
    # interpolation_config = data_module.interpolation_config
    # c = interpolation_config.get("condition")  # unormalized condition to evaluate
    # x1_true = interpolation_config.get("field_true")  # true field
    # domain_interior_true = interpolation_config.get("domain_interior_true")
    # x1_pred = model.query(
    #     c,
    #     interpolation_config={
    #         "interpolate": True,
    #         "target_domain": interpolation_config.get("domain_full_true"),
    #     },
    # )

    # plt.figure()
    # plt.plot(domain_interior_true.ravel(), utils.t2n(x1_pred[0].mean(0)))
    # plt.plot(domain_interior_true, x1_true[0])
    # plt.savefig("interpolation_low.png")
    # plt.close()

    return checkpointer.best_model_path


def train_residual_model(best_source_model_path):
    """train the residual model"""
    # Specs
    model = "high_fidelity"
    data_config = config.data.high_fidelity
    train_config = config.train.high_fidelity

    data_module = dataModule(config, model=model)

    model = residualFlow(
        config,
        nx=data_module.nx,
        nc=data_module.nc,
        nd=data_module.nd,
        domain=data_module.domain,
        normalization_config_file=data_module.normalization_file,
        boundary_conditions=data_module.boundary_conditions,
        best_source_model_path=best_source_model_path,
    )
    checkpointer = utils.get_checkpointer(data_config.checkpoint_path)
    trainer = utils.get_trainer(
        checkpointer=checkpointer,
        logger_name=data_config.logger_name,
        train_config=train_config,
    )
    if train_config.mode == "train":
        # Train
        trainer.fit(model, data_module)
        best_model_path = checkpointer.best_model_path
    elif train_config.mode == "eval":
        # Load the checkpoint
        assert (
            data_config.load_checkpoint is not None
        ), "For eval mode, a checkpoint must be provided"
        assert (
            data_config.load_dataset is True
        ), "For using past checkpoint, the relevant dataset must be loaded"
        best_model_path = (
            data_config.checkpoint_path + "/" + data_config.load_checkpoint
        )
        assert osp.exists(best_model_path), "Config file unavailable (Typo?)"
    else:
        raise ValueError("Invalid mode")

    # Load the best model
    model = residualFlow.load_from_checkpoint(best_model_path)
    # Evaluate the validation dataset
    n_gen = config.plot.n_gen
    model.evaluate_dataset(
        data_module.val_set, plot=True, n_gen=n_gen
    )  # get the prediction

    # Query the model
    interpolation_config = data_module.interpolation_config
    c = interpolation_config.get("condition")  # unormalized condition to evaluate
    x1_true = interpolation_config.get("field_true")  # true field
    x1_pred = model.query(
        c,
        interpolation_config={
            "interpolate": True,
            "target_domain": interpolation_config.get("domain_full_true"),
        },
        n_gen=n_gen,
    )

    x1_high_save = utils.t2n(model.append_boundary_conditions(x1_true.to(model.device)))
    x1_high_pred_save = utils.t2n(
        model.append_boundary_conditions(x1_pred.view(-1, model.nx)).view(
            x1_true.shape[0], n_gen, -1
        )
    )
    domain_true_save = utils.t2n(interpolation_config.get("domain_full_true"))

    if config.data.MF2M:
        x1_low = interpolation_config.get("field_coarse")
        x1_pred_low = model.source_model.query(
            c,
            interpolation_config={
                "interpolate": False,
                "target_domain": interpolation_config.get("domain_full_true"),
            },
            n_gen=n_gen,
        )
        x1_low_save = utils.t2n(
            model.source_model.append_boundary_conditions(x1_low.to(model.device))
        )
        x1_low_pred_save = utils.t2n(
            model.source_model.append_boundary_conditions(
                x1_pred_low.view(-1, model.source_model.nx)
            )
        ).reshape(x1_low.shape[0], n_gen, -1)

        domain_coarse_save = utils.t2n(interpolation_config.get("domain_full_coarse"))

        save_data = {}
        save_data["MF2M"] = {
            "true": x1_high_save,
            "pred": x1_high_pred_save,
            "domain": domain_true_save,
        }
        save_data["Low"] = {
            "true": x1_low_save,
            "pred": x1_low_pred_save,
            "domain": domain_coarse_save,
        }

        np.save("Poisson_MF2M.npy", save_data)

    else:
        save_data = {}
        save_data["High"] = {
            "true": x1_high_save,
            "pred": x1_high_pred_save,
            "domain": domain_true_save,
        }
        np.save("Poisson_High.npy", save_data)


def compare_fidelities():
    """Function compares the fidelities"""
    modes = [2, 5, 8]
    low_field_list = []
    high_field_list = []
    for ii in modes:
        config_low = config.data.low_fidelity
        poisson_low = Poisson(config_low, modes=ii)  # Solve the equations
        poisson_low.compute_dataset()
        low_field, low_condition = poisson_low.field, poisson_low.condition
        low_field = torch.cat(
            [
                0.0 * torch.ones(low_field.shape[0], 1),
                low_field,
                0.0 * torch.ones(low_field.shape[0], 1),
            ],
            dim=-1,
        )
        low_domain = poisson_low.domain_full

        config_high = config.data.high_fidelity
        high_field, high_domain, _ = poisson_low.get_true_solution(
            low_condition[0], n_pts=config_high.n_pts
        )
        high_field = utils.n2t(high_field).view(1, -1)

        low_field_list.append(low_field)
        high_field_list.append(high_field)

    fig, axs = plt.subplots(1, len(modes), figsize=(15, 5))
    axs = axs.ravel()
    for ii in range(len(modes)):
        axs[ii].plot(
            utils.t2n(low_domain),
            utils.t2n(low_field_list[ii][0]),
            label="Low Fidelity",
            color="red",
            marker="o",
        )
        axs[ii].plot(
            utils.t2n(high_domain),
            utils.t2n(high_field_list[ii][0]),
            label="High Fidelity",
            color="blue",
        )
        axs[ii].grid()
        axs[ii].set_xlabel(r"$x$")
        if ii == 0:
            axs[ii].set_ylabel(r"$u(x;\zeta)$")
            axs[ii].legend(loc="upper right")
        axs[ii].set_title(f"Modes, $M$: {modes[ii]}")
        axs[ii].set_ylim([-0.05, 0.05])
    plt.tight_layout()
    plt.savefig("fidelity_comparison.png")
    plt.close()


if __name__ == "__main__":
    # Compare the fidelities
    # compare_fidelities()

    # train the LF model
    if config.data.MF2M:
        assert (
            config.data.high_fidelity.load_dataset is True
        ), "Must be evaluated on the same dataset. Run without MF2M flag, then reuse"
        best_source_model_path = train_source_model()  # Train low fidelity model
        # best_source_model_path = (
        #     "experiments/mfFlow/Poisson/lowFidelity/"
        #     "checkpoints/model-epoch=2805-val_loss=0.02.ckpt"
        # )
    else:
        best_source_model_path = None  # Do not use Low fidelity model

    # train the HF model
    train_residual_model(best_source_model_path)
