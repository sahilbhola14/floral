import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as L
import argparse
import utils.utils as utils
import torch.nn as nn

from src.flow import Flow
from torch.utils.data import DataLoader, TensorDataset
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

torch.set_float32_matmul_precision("medium")  # for tensor cores


# DataModule
class dataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Load Low Fidelity Data
        utils.check_path(config.data.low_fidelity.data_path)
        self.low_data = np.load(config.data.low_fidelity.data_path, allow_pickle=True)
        # Load High Fidelity Data
        utils.check_path(config.data.high_fidelity.data_path)
        self.high_data = np.load(config.data.high_fidelity.data_path, allow_pickle=True)
        # Config
        self.data_config = self.config.data.high_fidelity
        self.loader_config = self.config.dataloader
        self.nx = self.high_data.get("y_high").shape[1]
        self.nc = self.high_data.get("features").shape[1]
        self.nd = self.high_data.get("x_high").shape[1]

    def setup(self, stage=None):
        print(f"Setting up data module for stage: {stage}")
        if stage == "eval":

            # Load the dataset
            utils.check_path("train_set.pt")
            utils.check_path("val_set.pt")

            self.train_set = torch.load("train_set.pt")
            self.val_set = torch.load("val_set.pt")

            # Load the test config
            utils.check_path("test_config.npz")
            self.test_config = np.load("test_config.npz", allow_pickle=True)

        else:
            # Create the dataset

            # Field
            high_field = utils.n2t(self.high_data.get("y_high"))
            high_field_at_domain = utils.n2t(self.high_data.get("y_high_at_domain"))

            low_field = utils.n2t(self.low_data.get("y_low_at_x_high"))
            low_field_at_domain_high = utils.n2t(
                self.low_data.get("y_low_at_domain_high")
            )

            domain = utils.n2t(self.high_data.get("x_high"))

            if self.config.data.corrFlow:
                # Train the residual
                field = high_field - low_field
            else:
                # Train the full field
                field = high_field

            # Condition
            condition = utils.n2t(self.high_data.get("features"))

            # Create subsets for training and validation
            assert self.data_config.n_samples <= len(
                field
            ), "Not enough samples in the dataset"
            field_sub = field[: self.data_config.n_samples]
            condition_sub = condition[: self.data_config.n_samples]
            domain_sub = domain[: self.data_config.n_samples]

            # Split for training and validation
            n_train = int(self.data_config.n_samples * self.loader_config.train_ratio)
            field_train, field_val = field_sub[:n_train], field_sub[n_train:]
            condition_train, condition_val = (
                condition_sub[:n_train],
                condition_sub[n_train:],
            )
            domain_train, domain_val = domain_sub[:n_train], domain_sub[n_train:]

            # Normalize the data
            field_mean, field_std = field_train.mean(0), field_train.std(0)
            condition_mean, condition_std = (
                condition_train.mean(0),
                condition_train.std(0),
            )

            field_train = (
                field_train - field_mean
            ) / field_std  # Normalize train field
            field_val = (field_val - field_mean) / field_std  # Normalize val field

            condition_train = (
                condition_train - condition_mean
            ) / condition_std  # Normalize train condition
            condition_val = (
                condition_val - condition_mean
            ) / condition_std  # Normalize val condition

            # Create Testing Config
            self.test_config = {}
            high_field_at_domain_val = high_field_at_domain[n_train:]
            low_field_at_domain_high_val = low_field_at_domain_high[n_train:]

            self.test_config["field"] = {
                "high": high_field_at_domain_val,
                "low": low_field_at_domain_high_val,
            }

            self.test_config["condition"] = condition_val

            self.test_config["domain"] = {
                "high": utils.n2t(self.high_data.get("domain")).view(-1, 1),
                "low": utils.n2t(self.low_data.get("domain")).view(-1, 1),
            }

            self.test_config["stats"] = {
                "field": {"mean": field_mean, "std": field_std},
                "condition": {"mean": condition_mean, "std": condition_std},
            }

            # Save Test Config
            np.savez("test_config.npz", **self.test_config)

            # Dataset
            self.train_set = TensorDataset(field_train, condition_train, domain_train)
            self.val_set = TensorDataset(field_val, condition_val, domain_val)

            # Save the dataset
            torch.save(self.train_set, "train_set.pt")
            torch.save(self.val_set, "val_set.pt")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=256, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=256, shuffle=False)


# ResFlow
class ResFlow(Flow, L.LightningModule):
    def __init__(self, config: dict, nx: int, nc: int, nd: int):
        super(ResFlow, self).__init__()
        self.save_hyperparameters()
        self.nx = nx
        self.nc = nc
        self.nd = nd

        # Config
        self.flow_config = config.flow
        self.sig_min = self.flow_config.sig_min
        self.time_emb_freq = self.flow_config.time_emb_freq
        self.latent_dim = self.flow_config.latent_dim

        # State embedding
        self.state_encoder = nn.Sequential(
            nn.Linear(self.nx + 2 * self.time_emb_freq, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, self.latent_dim),
        )

        # Conditional embedding
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.nc + 2 * self.time_emb_freq, 32),
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
        num_centers = self.flow_config.num_centers
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
        enc_time = self.time_embedding(t, self.time_emb_freq)
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


# Plot predictions
def plot_predictions(best_model_path, config, data_module):
    """Plot the predictions of the model"""

    # Load the best model
    model = ResFlow.load_from_checkpoint(best_model_path)
    # Set to eval mode
    model.eval()

    fig, axs = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    axs = axs.ravel()

    # Query the model on high fidelity domain
    d_eval = data_module.test_config["domain"]["high"]
    # dx = (d_eval[1] - d_eval[0]).item()

    for ii in range(len(axs) // 2):
        # Get the condition
        c_eval = data_module.test_config["condition"][ii].view(1, -1)
        # Get the true prediciton on domain
        high_field = data_module.test_config["field"]["high"][ii]
        # Get the low fidelity prediction
        low_field = data_module.test_config["field"]["low"][ii]
        # Get the model prediction
        pred_field = model.interpolate(c_eval, d_eval).squeeze(-1).T.detach().to("cpu")
        # Get the true prediction
        if config.data.corrFlow:
            # Denormalize
            pred_residual = (
                pred_field * data_module.test_config["stats"]["field"]["std"]
            ) + data_module.test_config["stats"]["field"]["mean"]

            pred_field = pred_residual + low_field

        # True Residual
        true_residual = high_field - low_field

        # Compute the Integral quantity
        # integral_field_true = comp_integral_field(utils.t2n(high_field), dx)
        # integral_field_pred = comp_integral_field(utils.t2n(pred_field), dx)
        # integral_field_pred_mean, integral_field_pred_std = integral_field_pred.mean(
        #     0
        # ), integral_field_pred.std(0)

        integral_field_true = 0.0
        integral_field_pred = 0.0
        integral_field_pred_mean, integral_field_pred_std = integral_field_pred.mean(
            0
        ), integral_field_pred.std(0)

        # Mean prediction
        mean_pred = utils.t2n(pred_field.mean(0))
        std_pred = utils.t2n(pred_field.std(0))

        # Field Plot
        axs[ii].plot(
            utils.t2n(d_eval),
            utils.t2n(high_field),
            label="High fidelity",
            color="blue",
        )
        axs[ii].plot(
            utils.t2n(d_eval), utils.t2n(low_field), label="Low fiedelity", color="red"
        )
        axs[ii].plot(utils.t2n(d_eval), mean_pred, label="corrFlow", color="green")
        axs[ii].fill_between(
            utils.t2n(d_eval).ravel(),
            mean_pred - std_pred,
            mean_pred + std_pred,
            alpha=0.2,
            color="green",
        )
        axs[ii].set_xlabel(r"x")
        axs[ii].set_ylabel(r"u(x,f(x))")
        title = r"$q$: {:.3f} $\vert$ ".format(
            integral_field_true.item()
        ) + r"$\hat{{q}}$: {:.3f} $\pm$ {:.3f}".format(
            integral_field_pred_mean.item(), integral_field_pred_std.item()
        )
        axs[ii].set_title(title)
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


if __name__ == "__main__":
    # DataModule
    data_module = dataModule(config)
    data_module.setup(config.train.stage)

    # Model
    model = ResFlow(config, nx=data_module.nx, nc=data_module.nc, nd=data_module.nd)

    # checkpointer
    checkpointer = utils.get_checkpointer(
        config.data.high_fidelity.checkpoint_save_path
    )

    # Trainer
    trainer = utils.get_trainer(
        checkpointer=checkpointer,
        logger_name=config.data.high_fidelity.logger_name,
        train_config=config.train,
    )

    # Train
    if config.train.stage == "train":
        # Train the model
        trainer.fit(model, data_module)
        # Load the best model
        best_model_path = checkpointer.best_model_path
    elif config.train.stage == "eval":
        print("Skipping training")
        utils.check_path(config.data.high_fidelity.checkpoint_load_path)
        best_model_path = config.data.high_fidelity.checkpoint_load_path

    # Plot predictions
    plot_predictions(best_model_path, config, data_module)
