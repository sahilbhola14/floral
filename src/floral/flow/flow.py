import math
import torch
from abc import ABC, abstractmethod
from torchdiffeq import odeint
from floral.utils import printer


class Flow(ABC):
    """Base class for flow models."""

    def __init__(self, hp_config):
        """Initialize the flow model."""
        self.hp_config = dict(hp_config)
        self.stepper_config = {
            "learning_rate": self.hp_config["learning_rate"],
            "optimizer": self.hp_config["optimizer"].lower().strip(),
            "weight_decay": self.hp_config["weight_decay"],
            "scheduler": self.hp_config["scheduler"].lower().strip(),
            "exponential_scheduler_gamma": self.hp_config[
                "exponential_scheduler_gamma"
            ],
            "lr_scheduler_step": self.hp_config["lr_scheduler_step"],
            "lr_scheduler_gamma": self.hp_config["lr_scheduler_gamma"],
        }

    def configure_optimizers(self, verbose=False):
        """optimizer configuration"""
        learning_rate = self.stepper_config["learning_rate"]
        weight_decay = self.stepper_config["weight_decay"]

        # optimizer
        if self.stepper_config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            if verbose:
                printer(
                    f"Using Adam optimizer with lr={learning_rate}"
                    f" and weight_decay={weight_decay}"
                )
        elif self.stepper_config["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            if verbose:
                printer(
                    f"Using AdamW optimizer with lr={learning_rate}"
                    f" and weight_decay={weight_decay}"
                )
        elif self.stepper_config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            if verbose:
                printer(
                    f"Using SGD optimizer with lr={learning_rate}"
                    f" and weight_decay={weight_decay}"
                )
        else:
            raise ValueError(
                "Unsupported optimizer:" f"{self.stepper_config['optimizer']}"
            )

        # scheduler
        available_schedulers = ["exponential", "steplr", "none"]
        if self.stepper_config["scheduler"] == "exponential":
            gamma = self.stepper_config["exponential_scheduler_gamma"]
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

            stepper = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 10,
                },
            }

        elif self.stepper_config["scheduler"] == "steplr":

            step_size = self.stepper_config["lr_scheduler_step"]
            gamma = self.stepper_config["lr_scheduler_gamma"]
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )

            stepper = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 10,
                },
            }

        elif self.stepper_config["scheduler"] == "none":

            stepper = {"optimizer": optimizer}

        else:
            raise ValueError(
                f"Scheduler: {self.stepper_config['scheduler']} is unavailable."
                f"Choose from: {','.join(available_schedulers)}"
            )

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
        vt = self._evaluate_vector_field(psi, c, d, t)
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

    def time_embedding(
        self,
        t: torch.Tensor,
        n_freq: int,
        style: str = "nerf",  # options: "linear" or "nerf"
        t_min: float = 0.0,  # only used for nerf
        t_max: float = 1.0,  # only used for nerf
    ):
        """time embedding function"""

        assert t.ndim == 2 and t.shape[-1] == 1, "Invalid shape"
        device = getattr(self, "device", t.device)
        dtype = t.dtype

        if style == "nerf":
            # Normalize to [0,1]
            denom = max(t_max - t_min, 1e-12)
            t_use = (t - t_min) / denom
            k = torch.arange(n_freq, device=device, dtype=dtype)
            freqs = (2.0**k) * torch.pi  # π·2^k
            phase = t_use * freqs
        elif style == "linear":
            t_use = t
            freqs = 2 * torch.arange(n_freq, device=device, dtype=dtype) * torch.pi
            phase = t_use * freqs
        else:
            raise ValueError(f"Unknown style: {style}")

        time_embed = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        assert (
            time_embed.ndim == 2 and time_embed.shape[-1] == 2 * n_freq
        ), "incorrect time embedding"

        return time_embed

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

    def _evaluate_vector_field(
        self, x: torch.Tensor, c: torch.Tensor, d: torch.Tensor, t: torch.Tensor
    ):
        """evalute the vector field of the flow"""
        # Embedd the time
        time_embed = self.time_embedding(t, self.time_embed_freq)
        # Embedd the state
        state_embed = self.state_embedding(state=x, time_embed=time_embed)
        # Embedd the condition
        condition_embed = self.condition_embedding(condition=c, time_embed=time_embed)
        # Scaling
        state_embed *= 1.0 / math.sqrt(self.latent_dim)
        condition_embed *= 1.0 / math.sqrt(self.latent_dim)
        # Learnable fusion
        out = self.fusion_embedding(
            state_embed=state_embed,
            condition_embed=condition_embed,
            time_embed=time_embed,
        )
        # Embedd the domain
        out = self.domain_embedding(out, d)

        assert not torch.isnan(out).any(), "Vector field is NaN"

        return out

    def _wrapper(
        self, x: torch.Tensor, c: torch.Tensor, d: torch.Tensor, t: torch.Tensor
    ):
        """wrapper function"""
        batch_size = x.shape[0] * x.shape[1]
        x_eval = x.view(batch_size, -1)
        c_eval = c.view(batch_size, -1)
        d_eval = d.reshape(batch_size, -1)
        t_eval = t.repeat(batch_size, 1)
        with torch.no_grad():
            vt = self._evaluate_vector_field(x_eval, c_eval, d_eval, t_eval)
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
        max_n_gen_per_batch: int = 50,  # reduce this if c_eval is high dimensional
    ):
        self.eval()  # set to eval mode
        c_eval, d_eval = c_eval.to(self.device), d_eval.to(self.device)

        assert c_eval.ndim == 2, "c_eval should be 2D"
        assert d_eval.ndim == 2, "d_eval should be 2D"
        assert c_eval.shape[-1] == self.nc, "c_eval should have shape (batch_size, nc)"
        assert c_eval.shape[0] == 1, "c_eval should have shape (1, nc)"

        # check for Nan and inf
        assert not (
            torch.isnan(c_eval).any() or torch.isinf(c_eval).any()
        ), "Nans/Infs found in condition."
        assert not (
            torch.isnan(d_eval).any() or torch.isinf(d_eval).any()
        ), "Nans/Infs found in domain."

        # batch size
        batch_size = d_eval.shape[0]

        # results
        results = []

        # Split n_gen into smaller batches
        for n_start in range(0, n_gen, max_n_gen_per_batch):
            n_end = min(n_start + max_n_gen_per_batch, n_gen)
            current_n_gen = n_end - n_start

            # create batches
            c_batch = c_eval.unsqueeze(1).expand(batch_size, current_n_gen, -1)
            d_batch = d_eval.unsqueeze(1).expand(-1, current_n_gen, -1)

            # Sample initial condition
            x0 = self.sample_initial_condition(c_batch, batch_size, current_n_gen)

            # Sample time
            t = torch.linspace(0, 1, nT, device=self.device, dtype=x0.dtype)

            # RHS
            def rhs(t, x):
                vt = self._wrapper(x, c_batch, d_batch, t)
                return vt

            # Integrate
            with torch.no_grad():
                x1_chunk = odeint(rhs, x0, t, method=method, atol=atol, rtol=rtol)[-1]
            results.append(x1_chunk)

            # clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(results, dim=1)

    @abstractmethod
    def sample_base_density(self, x1: torch.Tensor, c: torch.Tensor):
        """sample the base density"""
        pass

    @abstractmethod
    def sample_initial_condition(self, c: torch.Tensor, batch_size: int, n_gen: int):
        """get the initial condition for the flow"""
        pass
