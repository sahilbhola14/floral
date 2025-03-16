import torch
import matplotlib.pyplot as plt
import utils.utils as utils

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

    def __wrapper(
        self, x: torch.Tensor, c: torch.Tensor, d: torch.Tensor, t: torch.Tensor
    ):
        """wrapper function"""
        t_batch = t.repeat(x.shape[0], 1)
        return self.evaluate_vector_field(x, c, d, t_batch)

    @torch.no_grad()
    def query(
        self,
        c: torch.Tensor,
        d: torch.Tensor,
        n_gen: int = 100,
        n_Tsteps: int = 100,
        method="dopri5",
        atol=1e-6,
        rtol=1e-6,
        x1_true: torch.Tensor = None,
    ):
        """generate the condition (c) and domain (d), sample"""
        assert c.shape[0] == 1, "Only one condition is allowed"
        assert d.shape[-1] == self.nd, "Domain shape is incorrect"
        c, d = c.to(self.device), d.to(self.device)
        # Create the batches
        c_batch = c.repeat(n_gen, 1)  # repeat the condition
        x0 = self.sample_initial_condition(c_batch)  # sample the initial condition
        t = torch.linspace(0, 1, n_Tsteps, device=self.device)  # time steps
        # rhs function

        def rhs(t, x):
            return self.__wrapper(x, c_batch, d, t)

        x1_hat = odeint(rhs, x0, t, method=method, atol=atol, rtol=rtol)[-1]

        mean_prediction = utils.t2n(x1_hat.mean(dim=0))
        std_prediction = utils.t2n(x1_hat.std(dim=0))

        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.plot(
            utils.t2n(d).ravel(),
            mean_prediction,
            label="Prediction",
            color="red",
            marker="o",
        )
        axs.fill_between(
            utils.t2n(d).ravel(),
            mean_prediction - 2 * std_prediction,
            mean_prediction + 2 * std_prediction,
            alpha=0.3,
            color="red",
        )
        if x1_true is not None:
            axs.plot(
                utils.t2n(d).ravel(),
                utils.t2n(x1_true).ravel(),
                label="True",
                color="k",
                marker="o",
            )
        axs.set_xlabel("x")
        axs.set_ylabel("u(x; f(x))")
        axs.legend()
        axs.grid(True)
        plt.tight_layout()
        plt.savefig("prediction.png")
        plt.close()

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
    def sample_initial_condition(self, c: torch.Tensor):
        """get the initial condition for the flow"""
        pass
