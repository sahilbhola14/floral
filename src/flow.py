import torch
from abc import ABC, abstractmethod


class Flow(ABC):
    def __init__(self, learning_rate: float, weight_decay: float):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        """optimizer configuration"""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
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

    def compute_loss(self, x1: torch.Tensor, c: torch.Tensor, d: torch.Tensor = None):
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
        vt = self.evaluate_vector_field(psi, c, d)
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

    @abstractmethod
    def sample_base_density(self, x1: torch.Tensor, c: torch.Tensor):
        """sample the base density"""
        pass

    @abstractmethod
    def evaluate_vector_field(
        self, x: torch.Tensor, c: torch.Tensor, d: torch.Tensor = None
    ):
        """evaluate the vector field"""
        pass
