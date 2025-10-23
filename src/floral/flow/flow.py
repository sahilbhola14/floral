# src/floral/flow/flow.py
import sys
import lightning as L
import wandb
import torch
from floral.utils import (
    printer,
    omega_to_dict,
    check_tensor_blowup,
    check_keys,
    deep_get,
)
from floral.archs import get_vector_field_operator
from floral.gp import get_gp_prior
from torchdiffeq import odeint


class Flow(L.LightningModule):
    """Multi-fidelity flow class"""

    def __init__(
        self,
        config: dict,
        hp_config: wandb.sdk.wandb_config.Config | dict,
        domain_dict: dict[str, torch.Tensor],
        shape_dict: dict[str, int],
    ):
        super(Flow, self).__init__()
        # convert
        self.config = config if isinstance(config, dict) else omega_to_dict(config)
        self.hp_config = (
            hp_config if isinstance(hp_config, dict) else omega_to_dict(hp_config)
        )
        assert isinstance(domain_dict, dict)
        for (k, v) in domain_dict.items():
            domain_dict[k] = v.tolist() if isinstance(v, torch.Tensor) else v
        # shape dict
        self.shape_dict = shape_dict

        # save hyperparameters
        self.save_hyperparameters(
            {
                "config": self.config,
                "hp_config": self.hp_config,
                "shape_dict": self.shape_dict,
                "domain_dict": domain_dict,
            }
        )

        # save domain buffer
        for k, v in domain_dict.items():
            # conver value to tensors
            v_tensor = torch.Tensor(v)
            self.register_buffer(f"{k}_domain", v_tensor, persistent=True)

        # extract flow config
        flow_config = self.config["flow"]
        self.sig_min = flow_config.get("sig_min", 1e-5)
        # build operator config
        operator_config = self._get_operator_config(flow_config=flow_config)
        # build the operator modules for the vector field
        self.vector_field = get_vector_field_operator(operator_config=operator_config)
        # build the prior (eval mode implicit)
        self.prior = get_gp_prior(prior_config=flow_config["prior"])

    def training_step(self, batch, batch_idx):
        """training step"""
        assert len(batch) == 3, "expected: (target_field, condition, LF_field)"
        target_field, condition, _ = batch
        # compute the loss
        loss = self._comp_loss(target_field=target_field, condition=condition)
        # log the training loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step"""
        assert len(batch) == 3, "expected: (target_field, condition, LF_field)"
        target_field, condition, _ = batch
        # compute the loss
        loss = self._comp_loss(target_field=target_field, condition=condition)
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

    def _get_operator_config(self, flow_config):
        """get the operator config"""
        operator_config = flow_config.get("operator")
        # check required keys in flow config
        required_keys = ["field"]
        required_sub_keys = ["hidden_channels", "modes"]
        check_keys(operator_config, required_keys)

        # add field details
        check_keys(operator_config["field"], required_sub_keys)
        operator_config["field"]["channels"] = deep_get(
            self.shape_dict, ["field", "channels"]
        )
        operator_config["field"]["ndim"] = deep_get(self.shape_dict, ["field", "ndim"])
        # add condition details
        operator_config["condition"] = {}
        operator_config["condition"]["channels"] = deep_get(
            self.shape_dict, ["condition", "channels"]
        )
        operator_config["condition"]["ndim"] = deep_get(
            self.shape_dict, ["condition", "ndim"]
        )

        return operator_config

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

    def _sample_prior_measure(self, batch_size: int):
        """sample prior measure"""
        # prepare flattened domain
        domain_eval = self.field_domain.flatten(start_dim=2).transpose(1, 2).squeeze(0)

        # field shape
        field_channels = deep_get(self.shape_dict, ["field", "channels"])
        field_dims = deep_get(self.shape_dict, ["field", "dims"])
        # sample prior
        prior_samples = self.prior.sample(
            domain=domain_eval,
            batch_size=batch_size,
            field_channels=field_channels,
            field_dims=field_dims,
        )

        assert prior_samples.shape == (batch_size, field_channels, *field_dims)

        return prior_samples

    def _sample_conditional_flow(
        self, target_field: torch.Tensor, prior: torch.Tensor, t: torch.Tensor
    ):
        """sample the conditional flow"""
        assert (
            prior.shape == target_field.shape
        ), "prior sample shape not same as field shape"
        # extract shape
        batch_size = target_field.shape[0]
        field_ndim = deep_get(self.shape_dict, ["field", "ndim"])
        # reshape t
        t_expand = t.view(batch_size, 1, *([1] * field_ndim))
        # sample noise (fresh draw from prior measure)
        noise = self._sample_prior_measure(batch_size)
        # sample conditional flow
        psi = (
            t_expand * target_field + (1.0 - t_expand) * prior
        ) + self.sig_min * noise

        assert (
            psi.shape == target_field.shape
        ), "incorrect conditional flow sample shape"
        return psi

    def _comp_conditional_flow_derivative(
        self,
        target_field: torch.Tensor,
        prior: torch.Tensor,
    ):
        """compute the derivative (time) of the conditional flow"""
        assert (
            prior.shape == target_field.shape
        ), "prior sample shape not same as field shape"
        psi_prime = target_field - prior
        assert (
            psi_prime.shape == target_field.shape
        ), "incorrect conditional flow derivative"
        return psi_prime

    def _comp_loss(self, target_field: torch.Tensor, condition: torch.Tensor):
        """compute the flow-matching loss"""
        # extract shape
        batch_size, field_channels, *field_dims = target_field.shape
        # sample time from U[0, 1]
        t = torch.rand(batch_size, 1, device=self.device)
        # sample prior measure
        prior = self._sample_prior_measure(batch_size=batch_size)
        # sample the conditional flow
        psi = self._sample_conditional_flow(target_field=target_field, prior=prior, t=t)
        # compute conditional flow derivative
        psi_prime = self._comp_conditional_flow_derivative(
            target_field=target_field, prior=prior
        )
        # compute the model vector field
        vt = self.vector_field(
            psi=psi,
            condition=condition,
            field_domain=self.field_domain,
            t=t,
        )
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
            sys.exit("Unused parameters")
        else:
            pass

    def on_after_backward(self, check: bool = False):
        """for debugging of unused parameters"""
        if check:
            self._check_unused_parameters()
        else:
            pass

    def load_state_dict(self, state_dict, strict=True):
        # drop PyTorch-internal metadata if present
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)

    def _wrapper(self, field: torch.Tensor, condition: torch.Tensor, t: torch.Tensor):
        # sizes
        batch_size = field.shape[0]
        n_gen = field.shape[1]

        field_channels = deep_get(self.shape_dict, ["field", "channels"])
        field_dims = deep_get(self.shape_dict, ["field", "dims"])
        condition_channels = deep_get(self.shape_dict, ["condition", "channels"])
        condition_dims = deep_get(self.shape_dict, ["condition", "dims"])

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
                field_domain=self.field_domain,
                t=t_eval,
            )
        return vt.view(batch_size, n_gen, field_channels, *field_dims)

    @torch.no_grad()
    def integrate_flow(
        self,
        condition: torch.Tensor,
        n_gen: int = 10,
        nT: int = 10,
        method: str = "dopri5",
        atol: float = 1.0e-4,
        rtol: float = 1.0e-4,
        **kwargs,
    ):
        """integrate the flow"""
        # extract sizes
        batch_size = condition.shape[0]
        field_channels = deep_get(self.shape_dict, ["field", "channels"])
        field_dims = deep_get(self.shape_dict, ["field", "dims"])
        condition_channels = deep_get(self.shape_dict, ["condition", "channels"])
        condition_dims = deep_get(self.shape_dict, ["condition", "dims"])

        # create condition batch
        condition_batch = (
            condition.unsqueeze(1)
            .expand(-1, n_gen, condition_channels, *condition_dims)
            .to(self.device)
        )
        # sample prior
        prior = self._sample_prior_measure(batch_size=batch_size * n_gen).view(
            batch_size, n_gen, field_channels, *field_dims
        )

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
