import numpy as np
import torch
import pytorch_lightning as L
import argparse
import utils.utils as utils
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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


class Poisson_scalar:
    """poisson equation scalar solver"""

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


class Poisson:
    """poisson equaiton solver"""

    def __init__(self, config, rhs=None):
        self.n_pts = config.get("n_pts")
        self.n_samples = config.get("n_samples")
        self.domain = torch.linspace(0, 1, self.n_pts).view(-1, 1)
        self.kernel_width = 0.05  # kernel width
        self.boundary_normalization = {"condition": False, "field": False}
        if rhs is None:
            rhs = self.generate_rhs()
        u = torch.zeros_like(rhs)
        for ii in range(self.n_samples):
            u[:, ii] = self.solve(rhs[:, ii])

        self.field = u.T
        self.condition = rhs.T

    def solve(self, rhs):
        """solve the system"""
        dx = 1.0 / (len(rhs) - 1)  # grid spacing
        main_diag = (-2.0 / dx**2) * torch.ones(len(rhs))
        sub_diag = (1.0 / dx**2) * torch.ones(len(rhs))
        sup_diag = (1.0 / dx**2) * torch.ones(len(rhs))
        # enforce boundary conditions
        main_diag[0] = main_diag[-1] = 1.0
        sub_diag[-1] = sup_diag[0] = 0.0
        rhs[0] = rhs[-1] = 0.0
        A = sp.diags(
            [
                sub_diag[1:],
                main_diag,
                sup_diag[:-1],
            ],
            [-1, 0, 1],
            format="csc",
        )
        u = spla.spsolve(A, rhs)
        return utils.n2t(u)

    def evaluate_kernel(self, grid):
        """kernel function"""
        return torch.exp(
            -(torch.abs(grid[:, 0] - grid[:, 1]) ** 2) / (2 * self.kernel_width**2)
        )

    def generate_rhs(self):
        """rhs function"""
        x = utils.t2n(self.domain).ravel()
        XX, YY = np.meshgrid(x, x)
        grid = utils.n2t(np.stack([XX.ravel(), YY.ravel()]).T)
        cov = self.evaluate_kernel(grid).reshape(XX.shape)
        cov_sqrt = torch.linalg.cholesky(cov + 1e-6 * torch.eye(cov.shape[0]))
        return torch.zeros(self.n_pts, 1) + cov_sqrt @ torch.randn(
            self.n_pts, self.n_samples
        )


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
    def __init__(self, config: dict, nx: int, nc: int, nd: int):
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

        # Skip
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

        # Output
        # self.out = nn.Sequential(
        #         nn.Linear(self.nx, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, self.nx),
        #         )

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
        # encode the domain
        # enc_domain = self.position_embedding(d)
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
        self.loader_config = self.config.dataloader
        self.poisson = Poisson(self.data_config)  # Solve the equations

        self.setup()

    def setup(self, stage=None):
        """setup the data"""
        field = self.poisson.field
        condition = self.poisson.condition
        domain = self.poisson.domain

        self.nx = field.shape[1]
        self.nc = condition.shape[1]
        self.nd = domain.shape[1]

        # split
        n_train = int(self.n_samples * self.loader_config.train_ratio)

        condition_train, field_train = condition[:n_train], field[:n_train]

        condition_val, field_val = condition[n_train:], field[n_train:]

        # Normalize data
        if self.poisson.boundary_normalization.get("condition"):
            condition_mean = condition_train.mean(0)
            condition_std = condition_train.std(0)
            condition_train = (condition_train - condition_mean) / condition_std
            condition_val = (condition_val - condition_mean) / condition_std
        else:
            condition_mean = condition_train[:, 1:-1].mean(0)
            condition_std = condition_train[:, 1:-1].std(0)
            condition_train[:, 1:-1] = (
                condition_train[:, 1:-1] - condition_mean
            ) / condition_std
            condition_val[:, 1:-1] = (
                condition_val[:, 1:-1] - condition_mean
            ) / condition_std

        if self.poisson.boundary_normalization.get("field"):
            field_mean = field_train.mean(0)
            field_std = field_train.std(0)
            field_train = (field_train - field_mean) / field_std
            field_val = (field_val - field_mean) / field_std
        else:
            field_mean = field_train[:, 1:-1].mean(0)
            field_std = field_train[:, 1:-1].std(0)
            field_train[:, 1:-1] = (field_train[:, 1:-1] - field_mean) / field_std
            field_val[:, 1:-1] = (field_val[:, 1:-1] - field_mean) / field_std

        # Create the datasets
        self.train_set = CustomDataset(field_train, condition_train, domain)
        self.val_set = CustomDataset(field_val, condition_val, domain)

        # Save
        # torch.save(self.train_set, "train_set.pt")
        # torch.save(self.val_set, "val_set.pt")

        self.train_set = torch.load("train_set.pt")
        self.val_set = torch.load("val_set.pt")

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


if __name__ == "__main__":
    # Low fidelity
    data_module = dataModule(config, model="low_fidelity")
    model = sourceFlow(
        config,
        nx=data_module.nx,
        nc=data_module.nc,
        nd=data_module.nd,
    )
    checkpointer = utils.get_checkpointer(config.data.low_fidelity.checkpoint_path)
    trainer = utils.get_trainer(
        checkpointer=checkpointer,
        logger_name=config.data.low_fidelity.logger_name,
        train_config=config.train,
    )
    # train the model
    trainer.fit(model, data_module)
    # load the best model
    model = sourceFlow.load_from_checkpoint(checkpointer.best_model_path)
    model.evaluate_dataset(data_module.val_set, plot=True)  # get the prediction
