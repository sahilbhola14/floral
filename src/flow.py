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

    def compute_loss(self, x1: torch.Tensor, c: torch.Tensor):
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
        vt = self.evaluate_vector_field(psi, c, t)
        # Compute the loss
        loss = torch.mean((vt - psi_prime) ** 2)
        return loss

    def training_step(self, batch, batch_idx):
        """training step"""
        x1, c = batch
        loss = self.compute_loss(x1, c)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step"""
        x1, c = batch
        loss = self.compute_loss(x1, c)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def time_embedding(self, t: torch.Tensor, n_freq: int):
        """time embedding function"""
        f = 2 * torch.arange(n_freq, device=self.device) * torch.pi
        return torch.cat([torch.sin(f * t), torch.cos(f * t)], dim=-1)

    def __wrapper(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor):
        """wrapper function"""
        batch_size = x.shape[0] * x.shape[1]
        x_eval = x.view(batch_size, -1)
        c_eval = c.view(batch_size, -1)
        t_eval = t.repeat(batch_size, 1)
        vt = self.evaluate_vector_field(x_eval, c_eval, t_eval)
        return vt.view(x.shape)

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        n_gen: int = 100,  # number of generated samples (per initial condition)
        nT: int = 100,  # number of time steps
        method="dopri5",
        atol=1e-4,
        rtol=1e-4,
        plot: bool = False,
    ):
        self.eval()  # set to eval mode

        x1_true, c_true = dataset.tensors
        x1_true, c_true = x1_true.to(self.device), c_true.to(self.device)
        # create the batches
        batch_size = len(x1_true)
        c_batch = c_true.unsqueeze(1).repeat(1, n_gen, 1)
        x0 = self.sample_initial_condition(c_true, batch_size=batch_size, n_gen=n_gen)

        # time grid
        t = torch.linspace(0, 1, nT, device=self.device)
        # rhs function

        def rhs(t, x):
            return self.__wrapper(x, c_batch, t)

        x1_pred = odeint(rhs, x0, t, method=method, atol=atol, rtol=rtol)[-1]

        if plot:
            idx_plot = torch.randperm(len(x1_true))[:12]
            fig, axs = plt.subplots(3, 4, figsize=(12, 9))
            axs = axs.flatten()
            for ii in range(len(idx_plot)):
                x1_pred_plot = utils.t2n(x1_pred[idx_plot[ii]])
                x1_true_plot = utils.t2n(x1_true[idx_plot[ii]]).ravel()
                mean_pred = x1_pred_plot.mean(axis=0)
                std_pred = x1_pred_plot.std(axis=0)
                axs[ii].plot(mean_pred, label="Pred", color="red")
                axs[ii].plot(x1_true_plot, label="True", color="k")
                axs[ii].fill_between(
                    range(len(mean_pred)),
                    mean_pred - std_pred,
                    mean_pred + std_pred,
                    alpha=0.3,
                    color="red",
                )
                axs[ii].grid()
                if ii == 0:
                    axs[ii].legend()
                if ii % 4 == 0:
                    axs[ii].set_ylabel("u(x)", labelpad=10)
                if ii // 4 == 2:
                    axs[ii].set_xlabel("x")
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
    def sample_initial_condition(self, c: torch.Tensor, batch_size: int, n_gen: int):
        """get the initial condition for the flow"""
        pass
