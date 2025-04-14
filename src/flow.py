import torch

# import utils.utils as utils

from abc import ABC, abstractmethod
from torchdiffeq import odeint


class Flow(ABC):
    def configure_optimizers(self):
        """optimizer configuration"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        stepper = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 10,
            },
        }
        return stepper

    def compute_conditional_flow(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ):
        """compute the conditional flow"""
        normal_samples = torch.randn_like(x0, device=self.device)
        return (t * x1 + (1.0 - t) * x0) + self.sig_min * normal_samples

    def compute_conditional_flow_derivative(self, x0: torch.Tensor, x1: torch.Tensor):
        """compute the derivative (time) of the conditional flow"""
        return x1 - x0

    def compute_loss(self, x1: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
        """compute the vector field regression loss"""
        # Sample time
        t = torch.rand(len(x1), 1, device=self.device)
        # Sample base density
        x0 = self.sample_base_density(x1, c)
        # Compute the conditional flow
        psi = self.compute_conditional_flow(x0, x1, t)
        # Compute the conditional flow derivative
        psi_prime = self.compute_conditional_flow_derivative(x0, x1)
        # Compute the vector field
        vt = self.evaluate_vector_field(psi, c, d, t)
        # Compute the loss
        loss = torch.mean((vt - psi_prime) ** 2)
        return loss

    def training_step(self, batch, batch_idx):
        """training step"""
        x1, c, d = batch
        loss = self.compute_loss(x1, c, d)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step"""
        x1, c, d = batch
        loss = self.compute_loss(x1, c, d)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def time_embedding(self, t: torch.Tensor, n_freq: int):
        """time embedding function"""
        f = 2 * torch.arange(n_freq, device=self.device) * torch.pi
        return torch.cat([torch.sin(f * t), torch.cos(f * t)], dim=-1)

    def position_embedding(self, d: torch.Tensor):
        """position embedding function"""
        f = torch.arange(len(d), device=self.device).view(-1, 1) / 10000
        return (f * d).sin()

    def normalize_data(self, x: torch.Tensor = None, c: torch.Tensor = None):
        """function normalizes the data"""
        condition_stats = self.normalization_config["condition"]
        field_stats = self.normalization_config["field"]

        if c is not None:
            condition_mean = condition_stats["mean"].to(self.device)
            condition_std = condition_stats["std"].to(self.device)
            c = (c - condition_mean) / condition_std

        if x is not None:
            field_mean = field_stats["mean"].to(self.device)
            field_std = field_stats["std"].to(self.device)
            x = (x - field_mean) / field_std

        return x, c

    def denormalize_data(self, x: torch.Tensor = None, c: torch.Tensor = None):
        """function denormalizes the data"""
        condition_stats = self.normalization_config["condition"]
        field_stats = self.normalization_config["field"]

        if c is not None:
            condition_mean = condition_stats["mean"].to(self.device)
            condition_std = condition_stats["std"].to(self.device)
            c = c * condition_std + condition_mean

        if x is not None:
            field_mean = field_stats["mean"].to(self.device)
            field_std = field_stats["std"].to(self.device)
            x = x * field_std + field_mean

        return x, c

    def __wrapper(
        self, x: torch.Tensor, c: torch.Tensor, d: torch.Tensor, t: torch.Tensor
    ):
        """wrapper function"""
        batch_size = x.shape[0] * x.shape[1]
        x_eval = x.view(batch_size, -1)
        c_eval = c.view(batch_size, -1)
        d_eval = d.view(batch_size, -1)
        t_eval = t.repeat(batch_size, 1)
        vt = self.evaluate_vector_field(x_eval, c_eval, d_eval, t_eval)
        return vt.view(x.shape)

    @torch.no_grad()
    def interpolate(
        self,
        c_eval: torch.Tensor,
        d_eval: torch.Tensor,
        n_gen: int = 200,  # number of generated samples (per initial condition)
        nT: int = 100,  # number of time steps
        method="dopri5",
        atol=1e-4,
        rtol=1e-4,
        plot: bool = False,
    ):
        self.eval()  # set to eval mode
        c_eval, d_eval = c_eval.to(self.device), d_eval.to(self.device)
        assert c_eval.ndim == 2, "c_eval should be 2D"
        assert d_eval.ndim == 2, "d_eval should be 2D"
        assert c_eval.shape[-1] == self.nc, "c_eval should have shape (batch_size, nc)"
        assert c_eval.shape[0] == 1, "c_eval should have shape (1, nc)"
        # Create batches
        batch_size = d_eval.shape[0]
        c_batch = c_eval.unsqueeze(1).repeat(batch_size, n_gen, 1)
        d_batch = d_eval.unsqueeze(1).repeat(1, n_gen, 1)
        # Sample initial condition
        x0 = self.sample_initial_condition(c_batch, batch_size, n_gen)
        # Sample time
        t = torch.linspace(0, 1, nT, device=self.device)

        # RHS

        def rhs(t, x):
            vt = self.__wrapper(x, c_batch, d_batch, t)
            return vt

        # Integrate
        x1 = odeint(rhs, x0, t, method=method, atol=atol, rtol=rtol)[-1]

        return x1

    @abstractmethod
    def sample_base_density(self, x1: torch.Tensor, c: torch.Tensor):
        """sample the base density"""
        pass

    @abstractmethod
    def evaluate_vector_field(
        self, x: torch.Tensor, c: torch.Tensor, d: torch.Tensor, t: torch.Tensor
    ):
        """evaluate the vector field"""
        pass

    @abstractmethod
    def sample_initial_condition(self, c: torch.Tensor, batch_size: int, n_gen: int):
        """get the initial condition for the flow"""
        pass

    @abstractmethod
    def append_boundary_conditions(self, x: torch.Tensor):
        """append the boundary conditions to the data"""
        pass

    @abstractmethod
    def remove_boundary_conditions(self, x: torch.Tensor):
        """remove the boundary conditions to the data"""
        pass
