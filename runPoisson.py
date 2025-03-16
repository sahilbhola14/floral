import numpy as np
import torch
import pytorch_lightning as L
import argparse
import utils.utils as utils
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch.nn as nn

from src.archs.encoding import FiLM
from src.flow import Flow
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

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


class Poisson:
    """poisson equaiton solver"""

    def __init__(self, config, rhs=None):
        self.n_pts = config.get("n_pts")
        self.n_samples = config.get("n_samples")
        self.domain = torch.linspace(0, 1, self.n_pts).view(-1, 1)
        self.kernel_width = 0.05  # kernel width
        if rhs is None:
            rhs = self.generate_rhs()
        u = torch.zeros_like(rhs)
        for ii in range(self.n_samples):
            u[:, ii] = self.solve(rhs[:, ii])

        self.u = u.T
        self.rhs = rhs.T

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
    def __init__(self, data, condition, domain):
        self.data = data
        self.condition = condition
        self.domain = domain

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.condition[idx], self.domain


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
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
        )

        self.condition_encoder = nn.Sequential(
            nn.Linear(self.nc + 2 * self.n_freq, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim),
        )

        self.skip = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.film = FiLM(in_dim=self.nd, out_dim=self.latent_dim)  # FiLM

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
        # encode the domain
        skip = self.film(skip, d)

        return x + c + skip

    def sample_initial_condition(self, c: torch.Tensor):
        """sample the initial condition"""
        x0 = torch.randn_like(c, device=self.device)
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
        self.nx = self.data_config.n_pts  # field is defined on the domain
        self.nc = self.nx  # dimension of the conditional information
        self.nd = 1  # 1D data
        self.loader_config = self.config.dataloader

        self.poisson = Poisson(self.data_config)  # Solve the equations

    def setup(self, stage):
        """setup the data"""
        u = self.poisson.u
        rhs = self.poisson.rhs
        domain = self.poisson.domain

        # split
        n_train = int(len(u) * self.loader_config.train_ratio)
        rhs_train, u_train = rhs[:n_train], u[:n_train]
        rhs_val, u_val = rhs[n_train:], u[n_train:]

        # compute the stats
        mean_rhs, std_rhs = rhs_train[:, 1:-1].mean(), rhs_train[:, 1:-1].std()
        mean_u, std_u = u_train[:, 1:-1].mean(), u_train[:, 1:-1].std()

        # normalize
        u_train[:, 1:-1] = (u_train[:, 1:-1] - mean_u) / std_u
        u_val[:, 1:-1] = (u_val[:, 1:-1] - mean_u) / std_u
        rhs_train[:, 1:-1] = (rhs_train[:, 1:-1] - mean_rhs) / std_rhs
        rhs_val[:, 1:-1] = (rhs_val[:, 1:-1] - mean_rhs) / std_rhs

        # Create the datasets
        self.train_set = CustomDataset(u_train, rhs_train, domain)
        self.val_set = CustomDataset(u_train, rhs_train, domain)

    def collate_fn(self, batch):
        """Function for pre processing the batch"""
        data, condition, domain = zip(*batch)
        data = torch.stack(data)
        condition = torch.stack(condition)
        domain = domain[0]
        return data, condition, domain

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
    x, c, d = data_module.val_set[0]  # get the first sample
    model.query(c.view(1, -1), d, x1_true=x)  # get the prediction
