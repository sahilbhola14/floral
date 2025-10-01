# src/floral/flow/flow.py
import lightning as L
import wandb
import torch
from floral.utils import printer, omega_to_dict, make_grid, check_tensor_blowup
from floral.archs import get_operator_modules
from floral.gp import get_gp_prior
from torchdiffeq import odeint


class Flow(L.LightningModule):
    def __init__(
        self,
        config: dict,
        hp_config: wandb.sdk.wandb_config.Config | dict,
    ):
        super(Flow, self).__init__()
        self.config = config if isinstance(config, dict) else omega_to_dict(config)
        self.hp_config = (
            hp_config if isinstance(hp_config, dict) else omega_to_dict(hp_config)
        )
        # save hyperparameters
        self.save_hyperparameters(
            {
                "config": self.config,
                "hp_config": self.hp_config,
            }
        )
        # extract flow config
        flow_config = self.config["flow"]
        self.sig_min = flow_config.get("sig_min", 1e-5)
        # build the operator modules for the vector field
        self.vector_field = get_operator_modules(
            operator_config=flow_config["operator"]
        )
        # build the prior (eval mode implicit)
        self.prior = get_gp_prior(prior_config=flow_config["prior"])

    def training_step(self, batch, batch_idx):
        """training step"""
        assert len(batch) == 4, "expected: (target_field, condition, domain, LF_field)"
        target_field, condition, domain, _ = batch
        # compute the loss
        loss = self._comp_loss(
            target_field=target_field, condition=condition, domain=domain
        )
        # log the training loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step"""
        assert len(batch) == 4, "expected: (target_field, condition, domain, LF_field)"
        target_field, condition, domain, _ = batch
        # compute the loss
        loss = self._comp_loss(
            target_field=target_field, condition=condition, domain=domain
        )
        # log the validation loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self, verbose=False):
        """configure the optimizers
        Returns:
            stepper (dict):
                dict for the optimzer and scheduler
        """
        # get the stepper config
        stepper_config = self._get_stepper_config()
        # extract optimizer and scheduler
        optimizer = stepper_config.get("optimizer")
        scheduler = stepper_config.get("scheduler")
        # extract params
        weight_decay = stepper_config.get("weight_decay")
        learning_rate = stepper_config.get("learning_rate")
        # build optimizer
        if optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            if verbose:
                printer(
                    f"Using Adam optimizer with lr={learning_rate}"
                    f" and weight_decay={weight_decay}"
                )
        elif optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            if verbose:
                printer(
                    f"Using AdamW optimizer with lr={learning_rate}"
                    f" and weight_decay={weight_decay}"
                )
        elif optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            if verbose:
                printer(
                    f"Using SGD optimizer with lr={learning_rate}"
                    f" and weight_decay={weight_decay}"
                )
        else:
            raise ValueError("Unsupported optimizer:" f"{optimizer}")

        # build scheduler
        if scheduler == "exponential":
            gamma = stepper_config["exponential_scheduler_gamma"]
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

            stepper = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 10,
                },
            }

        elif scheduler == "steplr":
            step_size = stepper_config["lr_scheduler_step"]
            gamma = stepper_config["lr_scheduler_gamma"]
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

        elif scheduler == "none":

            stepper = {"optimizer": optimizer}

        else:
            raise ValueError(f"Unavailable scheduler: {scheduler}")

        return stepper

    def _get_stepper_config(self):
        """build the stepper config
        Returns:
            stepper_config (dict):
                stepper (optimizer + scheduler) configuration
        """
        # build stepper config
        stepper_config = {
            "learning_rate": self.hp_config.get("learning_rate", 1e-3),
            "optimizer": self.hp_config.get("optimizer", "adam").lower().strip(),
            "weight_decay": self.hp_config.get("weight_decay", 1e-4),
            "scheduler": self.hp_config.get("scheduler", "exponential").lower().strip(),
            "exponential_scheduler_gamma": self.hp_config.get(
                "exponential_scheduler_gamma", 0.99
            ),
            "lr_scheduler_step": self.hp_config.get("lr_scheduler_step", 5),
            "lr_scheduler_gamma": self.hp_config.get("lr_scheduler_gamma", 0.1),
        }
        # assert
        assert (
            stepper_config.get("learning_rate") > 0
        ), "learning rate cannot be negative"
        optimizer_list = ["adam", "adamw", "sgd"]
        assert (
            stepper_config.get("optimizer") in optimizer_list
        ), f"optimizer should be in {', '.join(optimizer_list)}"
        assert (
            stepper_config.get("weight_decay") >= 0
        ), "weight_decay must be greater than or equal to zero"
        scheduler_list = ["exponential", "steplr", "none"]
        assert (
            stepper_config.get("scheduler") in scheduler_list
        ), f"scheduler should be in {', '.join(scheduler_list)}"
        assert (
            stepper_config.get("exponential_scheduler_gamma") >= 0
        ), "exponential scheduler gamma must be positive"
        lr_scheduler_step = stepper_config.get("lr_scheduler_step")
        assert lr_scheduler_step >= 0 and isinstance(
            lr_scheduler_step, int
        ), "lr scheduler step must be positive integer"
        assert (
            stepper_config.get("lr_scheduler_gamma") >= 0
        ), "lr scheduler gamma must be positive"

        return stepper_config

    def _sample_prior_measure(self, n_samples: int, n_channels: int, dims: int):
        """sample prior measure"""
        # create query points based on the dims
        query_points = make_grid(dims).to(self.device)
        # sample prior
        prior_samples = self.prior.sample(
            x=query_points,
            dims=dims,
            n_samples=n_samples,
            n_channels=n_channels,
        )
        return prior_samples

    def _sample_conditional_flow(
        self, target_field: torch.Tensor, prior: torch.Tensor, t: torch.Tensor
    ):
        """sample the conditional flow"""
        assert (
            prior.shape == target_field.shape
        ), "prior sample shape not same as field shape"
        # extract shape
        batch_size, field_channels, *field_dims = target_field.shape
        # reshape t
        t_expand = t.view(batch_size, 1, *([1] * len(field_dims)))
        # sample noise (fresh draw from priro measure)
        noise = self._sample_prior_measure(
            n_samples=batch_size, n_channels=field_channels, dims=field_dims
        )
        return (
            t_expand * target_field + (1.0 - t_expand) * prior
        ) + self.sig_min * noise

    def _comp_conditional_flow_derivative(
        self,
        target_field: torch.Tensor,
        prior: torch.Tensor,
    ):
        """compute the derivative (time) of the conditional flow"""
        assert (
            prior.shape == target_field.shape
        ), "prior sample shape not same as field shape"
        return target_field - prior

    def _comp_loss(
        self, target_field: torch.Tensor, condition: torch.Tensor, domain: torch.Tensor
    ):
        """compute the flow-matching loss"""
        # extract shape
        batch_size, field_channels, *field_dims = target_field.shape
        # sample time from U[0, 1]
        t = torch.rand(batch_size, 1, device=self.device)
        # sample prior measure
        prior = self._sample_prior_measure(
            n_samples=batch_size, n_channels=field_channels, dims=field_dims
        )
        # sample the conditional flow
        psi = self._sample_conditional_flow(target_field=target_field, prior=prior, t=t)
        # compute conditional flow derivative
        psi_prime = self._comp_conditional_flow_derivative(
            target_field=target_field, prior=prior
        )
        # compute the model vector field
        vt = self.vector_field(psi=psi, condition=condition, t=t)

        assert vt.shape == psi_prime.shape, "incorrect target and model vector field"

        # Compute the loss
        loss = torch.mean((vt - psi_prime) ** 2)
        return loss

    def _check_unused_parameters(self):
        """Check which parameters or buffers have no gradients after backward."""
        unused = []
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is None:
                unused.append(("param", name, tuple(param.shape)))
        for name, buf in self.named_buffers():
            # Buffers usually don't require grad, but check if they do
            if getattr(buf, "requires_grad", False) and buf.grad is None:
                unused.append(("buffer", name, tuple(buf.shape)))
        if unused:
            print("\n[Unused parameters/buffers detected]")
            for kind, name, shape in unused:
                print(f" - {kind}: {name}, shape={shape}")
        else:
            print("\n[All parameters that require grad received gradients]")

    # def on_after_backward(self):
    #     """ for debugging of unused parameters """
    #     self._check_unused_parameters()

    def load_state_dict(self, state_dict, strict=True):
        # drop PyTorch-internal metadata if present
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)

    def _wrapper(self, field: torch.Tensor, condition: torch.Tensor, t: torch.Tensor):
        batch_size, n_gen, field_channels, *field_dims = field.shape
        batch_size, n_gen, condition_channels, *condition_dims = condition.shape
        # create eval field and condition
        field_eval = field.view(batch_size * n_gen, field_channels, *field_dims)
        condition_eval = condition.reshape(
            batch_size * n_gen, condition_channels, *condition_dims
        )
        t_eval = (
            torch.ones(batch_size * n_gen, 1, device=self.device) * t
        )  # (batch_size, 1)
        with torch.no_grad():
            vt = self.vector_field(
                psi=field_eval,
                condition=condition_eval,
                t=t_eval,
            )

        return vt.view(batch_size, n_gen, field_channels, *field_dims)

    @torch.no_grad()
    def integrate_flow(
        self,
        condition: torch.Tensor,
        field_channels: int,
        field_dims: list,
        n_gen: int = 10,
        nT: int = 10,
        method: str = "dopri5",
        atol: float = 1.0e-4,
        rtol: float = 1.0e-4,
        **kwargs,
    ):
        """integrate the flow"""
        # create condition batch
        batch_size, condition_channels, *condition_dims = condition.shape
        condition_batch = condition.unsqueeze(1).expand(
            -1, n_gen, condition_channels, *condition_dims
        )
        # sample prior
        prior = self._sample_prior_measure(
            n_samples=batch_size * n_gen, n_channels=field_channels, dims=field_dims
        ).view(batch_size, n_gen, field_channels, *field_dims)

        prior = prior.to(self.device)

        # interation times
        t = torch.linspace(0, 1, nT).to(self.device)

        # RHS
        def _rhs(t, field):
            vt = self._wrapper(
                field=field,
                condition=condition_batch,
                t=t,
            )
            assert (
                vt.shape == field.shape
            ), f"incorrect vecotor field shape. Expected {field.shape}, got {vt.shape}"
            check_tensor_blowup(vt, "vt vector field")
            return vt

        # integrate
        with torch.no_grad():
            x1 = odeint(
                _rhs,
                prior,
                t,
                method=method,
                atol=atol,
                rtol=rtol,
            )[-1]

        assert x1.shape == (batch_size, n_gen, field_channels, *field_dims)

        return x1
