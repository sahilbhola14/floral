import numpy as np
import torch
import pytorch_lightning as L
import argparse
import utils.utils as utils
import os.path as osp

# import scipy.sparse as sp

# import scipy.sparse.linalg as spla
import torch.nn as nn

from scipy.sparse import diags

from scipy.sparse.linalg import spsolve
from src.flow import Flow
from torch.utils.data import DataLoader, Dataset
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


class Poisson_case_1:
    """poisson equation scalar solver
    Solve the differential equation:
    d²u/dx² + θ₁sin(θ₂x)u = 0
    with boundary conditions u(0) = 0, u(1) = 1
    """

    def __init__(self, config, rhs=None):
        self.n_pts = config.get("n_pts")
        self.n_samples = config.get("n_samples")
        self.condition = self.sample_theta()  # condition for the field
        self.field = torch.zeros(self.n_samples, self.n_pts)  # field
        self.boundary_normalization = {"condition": True, "field": False}

        for ii in range(self.n_samples):
            x, u = self.solve(
                self.condition[ii, 0], self.condition[ii, 1], N=self.n_pts - 2
            )
            self.field[ii] = utils.n2t(u)
        self.domain = utils.n2t(x).view(-1, 1)  # domain for the field

    def sample_theta(self):
        """Sample the model parameters"""
        return utils.n2t(
            np.array([np.random.uniform(0, 5, self.n_samples) for _ in range(2)])
        ).T

    def solve(self, theta1, theta2, N):
        """
        Solve the differential equation:
        d²u/dx² + θ₁sin(θ₂x)u = 0
        with boundary conditions u(0) = 0, u(1) = 1

        Parameters:
        theta1 : float
        Coefficient θ₁
        theta2 : float
        Coefficient θ₂
        N : int
        Number of interior points for discretization
        Returns:
        tuple: (x, u) where x is the grid points and u is the solution
        """

        # Create grid points (including boundaries)
        h = 1.0 / (N + 1)  # Step size
        x = np.linspace(0, 1, N + 2)

        # Create coefficient matrix for interior points
        # The equation becomes: (u[i+1] - 2u[i] + u[i-1])/h² + θ₁sin(θ₂x[i])u[i] = 0

        # Main diagonal: -2/h² + θ₁sin(θ₂x)
        main_diag = -2.0 / (h**2) + theta1 * np.sin(theta2 * x[1:-1])

        # Off diagonals: 1/h²
        off_diag = np.ones(N - 1) / (h**2)

        # Create sparse matrix
        diagonals = [main_diag, off_diag, off_diag]
        A = diags(diagonals, [0, 1, -1], shape=(N, N)).tocsr()

        # Create right-hand side vector
        b = np.zeros(N)
        b[-1] = -1.0 / (h**2)  # Due to boundary condition u(1) = 1

        # Solve the system
        u_interior = spsolve(A, b)

        # Combine with boundary conditions
        u = np.zeros(N + 2)
        u[1:-1] = u_interior
        u[-1] = 1.0  # u(1) = 1

        return x, u


class Poisson_case_2:
    """
    Solve the differential equation: d²u/dx² + f(x;θ) = 0
    with boundary conditions u(0) = 0, u(1) = 0,
    where f(x;θ) = sum_{i=1}^n θ_i sin(2πi x)
    """

    def __init__(self, config):
        self.n_pts = config.get("n_pts")
        self.n_samples = config.get("n_samples")
        self.n_modes = 5  # Number of modes for the forcing term
        self.L = 1.0  # domain length
        self.boundary_conditions = {
            "domain": [0.0, 1.0],
            "value": [0.0, 0.0],
        }  # Values of boundary conditions
        self.domain = torch.linspace(0, self.L, self.n_pts)[1:-1]  # interior domain
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
        x = self.domain  # interior points
        dx = self.L / (N + 1)
        A = (
            diags([1, -2, 1], [-1, 0, 1], shape=(N, N)) / dx**2
        )  # Discretized Laplacian
        f = self.get_rhs(x, theta)  # Forcing term
        u = spsolve(A, -f)  # Solve linear system
        return np.concatenate(([0], u, [0]))  # Apply boundary conditions


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
        normalization_config_file: str,
        boundary_conditions: dict,
    ):
        super(sourceFlow, self).__init__()
        self.save_hyperparameters()

        self.config = config  # config
        self.nx = nx  # dimension of the field
        self.nc = nc  # dimension of the conditional information
        self.nd = nd  # dimension of the domain
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
        self, x: torch.Tensor, c: torch.Tensor, d: torch.Tensor, t: torch.Tensor
    ):
        """evaluate the vector field"""
        # encod time
        enc_time = self.time_embedding(t, self.n_freq)
        # encode the state
        enc_state = self.state_encoder(torch.cat([x, enc_time], dim=-1))
        # encode the condition
        enc_cond = self.condition_encoder(torch.cat([c, enc_time], dim=-1))
        # skip connection
        skip = self.skip(enc_state + enc_cond)
        out = enc_state + enc_cond + skip
        # Encode the domain
        out = self.domain_encoder(out, d)

        return out

    def sample_initial_condition(self, c: torch.Tensor, batch_size: int, n_gen: int):
        """sample the initial condition"""
        x0 = torch.randn(batch_size, n_gen, self.nx, device=self.device)
        return x0

    def append_boundary_conditions(
        self, x: torch.Tensor = None, d: torch.Tensor = None
    ):
        """function appends the boundary conditon"""
        if x is not None:
            left_boundary = self.boundary_conditions.get("value")[0] * torch.ones(
                x.shape[0], 1, device=self.device
            )
            right_boundary = self.boundary_conditions.get("value")[1] * torch.ones(
                x.shape[0], 1, device=self.device
            )
            x = torch.cat([left_boundary, x, right_boundary], dim=-1)
        if d is not None:
            left_boundary = self.boundary_conditions.get("domain")[0] * torch.ones(
                1, 1, device=self.device
            )
            right_boundary = self.boundary_conditions.get("domain")[1] * torch.ones(
                1, 1, device=self.device
            )
            d = torch.cat([left_boundary, d, right_boundary], dim=0)

        return x, d


class residualFlow(Flow, L.LightningModule):
    """RESIDUAL Flow Model"""

    def __init__(
        self, config: dict, nx: int, nc: int, nd: int, best_source_model_path: str
    ):
        super(residualFlow, self).__init__()
        self.save_hyperparameters()

        self.config = config  # config
        self.nx = nx  # dimension of the field
        self.nc = nc  # dimension of the conditional information
        self.nd = nd  # dimension of the domain
        self.flow_config = self.config.flow
        self.sig_min = self.flow_config.sig_min
        self.latent_dim = self.flow_config.latent_dim
        self.n_freq = self.flow_config.time_emb_freq

        # source flow
        self.best_source_model_path = best_source_model_path
        self.source_model = self.initialize_source_flow_model()

        # State embedding
        self.state_encoder = nn.Sequential(
            nn.Linear(self.nx + 2 * self.n_freq, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.latent_dim),
        )

        # Conditional embedding
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.nc + 2 * self.n_freq, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.latent_dim),
        )

        # Skip connection
        self.skip = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.latent_dim),
        )

        # Domain encoder
        self.domain_encoder = RBFFiLM(in_dim=self.nd, out_dim=self.latent_dim)

    def initialize_source_flow_model(self):
        """Initialize the source flow model (set to eval mode)"""
        source_model = sourceFlow.load_from_checkpoint(self.best_source_model_path)
        for param in source_model.parameters():
            param.requires_grad = False
        return source_model

    def query_source_model(self, c_normalized):
        """Query the source model
        Args:
            c_normalized (torch.Tensor): normalized condition tensor
        """
        raise NotImplementedError

    def sample_base_density(self, x1: torch.Tensor, c: torch.Tensor):
        """sample the base density"""
        raise NotImplementedError

    def evaluate_vector_field(
        self, x: torch.Tensor, c: torch.Tensor, d: torch.Tensor, t: torch.Tensor
    ):
        """evaluate the vector field"""
        raise NotImplementedError

    def sample_initial_condition(self, c: torch.Tensor, batch_size: int, n_gen: int):
        """sample the initial condition"""
        raise NotImplementedError


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
        self.poisson = Poisson_case_2(self.data_config)  # Solve the equations

        self.nx = self.poisson.nx
        self.nc = self.poisson.nc
        self.nd = self.poisson.nd
        self.domain = self.poisson.domain

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

    def setup(self, stage=None):
        """setup the data"""
        if self.data_config.load_dataset is False:
            # Solve the poisson equation
            self.poisson.compute_dataset()

            # Extract the data
            field = self.poisson.field
            condition = self.poisson.condition
            domain = self.poisson.domain.view(-1, 1)

            # split
            n_train = int(self.n_samples * self.loader_config.train_ratio)
            condition_train, field_train = condition[:n_train], field[:n_train]
            condition_val, field_val = condition[n_train:], field[n_train:]

            # Compute the statstistics
            self.comp_dataset_stats(field_train, condition_train)

            # Normalize the data
            field_train, condition_train = self.normalize_dataset(
                field_train, condition_train
            )
            field_val, condition_val = self.normalize_dataset(field_val, condition_val)

            # Create the datasets
            self.train_set = CustomDataset(field_train, condition_train, domain)
            self.val_set = CustomDataset(field_val, condition_val, domain)

            # Save
            save_data_dict = {"train": self.train_set, "val": self.val_set}

            torch.save(save_data_dict, self.dataset_file)
            torch.save(self.normalization_config, self.normalization_file)

        else:
            data_dict = torch.load(self.dataset_file)
            self.train_set = data_dict.get("train")
            self.val_set = data_dict.get("val")

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
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        """val dataloader"""
        return DataLoader(
            self.val_set,
            batch_size=self.loader_config.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
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
        best_model_path = (
            data_config.checkpoint_path + "/" + data_config.load_checkpoint
        )
        assert osp.exists(best_model_path), "Config file unavailable (Typo?)"
    else:
        raise ValueError("Invalid mode")

    # Load the best model
    model = sourceFlow.load_from_checkpoint(best_model_path)
    # Plotting
    model.evaluate_dataset(
        data_module.val_set, plot=True, n_gen=2
    )  # get the prediction

    return checkpointer.best_model_path


if __name__ == "__main__":
    # train the LF model
    best_source_model_path = train_source_model()
    # best_source_model_path = ""

    # train the HF model
    # train_residual_model(best_source_model_path)
