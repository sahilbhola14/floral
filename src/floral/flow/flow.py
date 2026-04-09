# src/floral/flow/flow.py
"""
Flow mathching operator
"""
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
from collections.abc import Mapping
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from torchdiffeq import odeint


@dataclass
class FlowConfig:
    """Default config for Flow with override support."""

    # config defaults
    config: dict = field(
        default_factory=lambda: {
            "floral": False,
            "flow": {
                "sig_min": 1e-5,
                "operator": {
                    "method": "FiLMFNO",
                    "field": {
                        "hidden_channels": 64,
                        "lifting_channel_ratio": 4,
                        "projection_channel_ratio": 4,
                        "n_layers": 4,
                        "modes": 64,
                    },
                },
                "prior": {
                    "lengthscale": 1e-3,
                    "outputscale": 1.0,
                    "confidence": 0.6,
                },
            },
        }
    )

    # hp config defaults
    hp_config: dict = field(
        default_factory=lambda: {
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "weight_decay": 1e-4,
            "scheduler": "exponential",
            "exponential_scheduler_gamma": 0.99,
            "lr_scheduler_step": 5,
            "lr_scheduler_gamma": 0.1,
        }
    )

    @staticmethod
    def _to_dict_config(value):
        if value is None:
            return OmegaConf.create({})
        if isinstance(value, DictConfig):
            return value
        if isinstance(value, Mapping):
            return OmegaConf.create(dict(value))
        if hasattr(value, "items"):
            return OmegaConf.create({k: v for k, v in value.items()})
        raise TypeError(f"Unsupported config type: {type(value).__name__}")

    def resolve(self, config=None, hp_config=None):
        merged_config = OmegaConf.merge(
            OmegaConf.create(self.config),
            self._to_dict_config(config),
        )
        merged_hp_config = OmegaConf.merge(
            OmegaConf.create(self.hp_config),
            self._to_dict_config(hp_config),
        )
        return merged_config, merged_hp_config


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
        # merge defaults with user-provided runtime config/hp_config
        config_defaults = FlowConfig()
        config, hp_config = config_defaults.resolve(config=config, hp_config=hp_config)
        # convert
        self.config = config if isinstance(config, dict) else omega_to_dict(config)
        self.hp_config = (
            hp_config if isinstance(hp_config, dict) else omega_to_dict(hp_config)
        )
        assert isinstance(domain_dict, dict)
        for k, v in domain_dict.items():
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
            # convert value to tensors
            v_tensor = torch.Tensor(v)
            self.register_buffer(f"{k}_domain", v_tensor, persistent=True)

        # floral
        self.floral = self.config.get("floral")

        # extract flow config
        self.flow_config = self.config.get("flow")
        self.sig_min = self.flow_config.get("sig_min")
        # check train resolution
        self._check_train_resolution()
        # get slice operator
        self.slice_op = self._get_slice_operator()
        # build operator config
        operator_config = self._get_operator_config()
        # build the operator modules for the vector field
        self.vector_field = get_vector_field_operator(
            operator_config=operator_config, floral=self.floral
        )
        # build the prior (eval mode implicit)
        self.prior_config = self.flow_config.get("prior")
        # in case of residual, prior is scaled by 2.0 as r_0 = x_0 - \hat{x}_0, where
        # both rhs variables are GP(0, k(x, x^\prime)).
        prior_scale = 2.0 if self.floral else 1.0
        self.prior = get_gp_prior(
            lengthscale=self.prior_config.get("lengthscale"),
            outputscale=self.prior_config.get("outputscale") * prior_scale,
            confidence=0.0,  # no bias
        )
        # noise
        self.noise = get_gp_prior(
            lengthscale=self.prior_config.get("lengthscale"),
            outputscale=self.prior_config.get("outputscale"),
            confidence=0.0,  # no bias
        )

    def _check_train_resolution(self):
        """check the training resolution"""
        train_res = deep_get(self.config, ["train", "train_res"])
        field_dims = deep_get(self.shape_dict, ["field", "dims"])

        assert train_res == "Full" or isinstance(
            train_res, int
        ), "train resolution can be Full or an integer"
        if train_res == "Full":
            # set the train res to the field dims
            train_res = deep_get(self.shape_dict, ["field", "dims"])
        else:
            assert all(
                [train_res <= dim for dim in field_dims]
            ), "train resolution cannot be greater that field resolution"
            train_res = [train_res] * deep_get(self.shape_dict, ["field", "ndim"])
        # check field dims
        field_dims = deep_get(self.shape_dict, ["field", "dims"])
        assert all(
            [dim % res == 0 for dim, res in zip(field_dims, train_res)]
        ), "full field dims must be multiple of train_res"
        # check conditions dims
        condition_dims = deep_get(self.shape_dict, ["condition", "dims"])
        assert all(
            [dim % res == 0 for dim, res in zip(condition_dims, train_res)]
        ), "full condition dims must be multiple of train_res"
        # check field domain dims
        field_domain_dims = deep_get(self.shape_dict, ["field_domain", "dims"])
        assert all(
            [dim % res == 0 for dim, res in zip(field_domain_dims, train_res)]
        ), "full condition dims must be multiple of train_res"
        # check condition domain dims
        condition_domain_dims = deep_get(self.shape_dict, ["condition_domain", "dims"])
        assert all(
            [dim % res == 0 for dim, res in zip(condition_domain_dims, train_res)]
        ), "full condition dims must be multiple of train_res"

    def _get_slice_operator(self):
        """get the slice"""
        train_res = deep_get(self.config, ["train", "train_res"])
        field_dims = deep_get(self.shape_dict, ["field", "dims"])
        assert train_res == "Full" or isinstance(
            train_res, int
        ), "train resolution can be Full or an integer"
        if train_res == "Full":
            # set the train res to the field dims
            train_res = field_dims
        else:
            train_res = [train_res] * deep_get(self.shape_dict, ["field", "ndim"])

        skips = [dim // res for dim, res in zip(field_dims, train_res)]
        slice_op = (slice(None),) * 2 + tuple(
            [slice(0, dim, skip) for dim, skip in zip(field_dims, skips)]
        )
        return slice_op

    def _log_losses(self, loss: torch.Tensor, stage: str):
        """log all losses with stage prefix"""
        log_kwargs = {"on_step": False, "on_epoch": True, "sync_dist": True}
        self.log(f"{stage}_loss", loss, prog_bar=True, **log_kwargs)

    def training_step(self, batch, batch_idx):
        """training step"""
        # extract the batch
        required_keys = ["target_field", "condition", "LF_field"]
        check_keys(batch, required_keys)
        target_field = batch.get("target_field")
        condition = batch.get("condition")
        LF_field = batch.get("LF_field")

        # compute the loss
        loss = self._comp_loss(
            target_field=target_field, condition=condition, LF_field=LF_field
        )

        # log the loss
        self._log_losses(loss, stage="train")

        return loss

    def validation_step(self, batch, batch_idx):
        """validation step"""
        # extract the batch
        required_keys = ["target_field", "condition", "LF_field"]
        check_keys(batch, required_keys)
        target_field = batch.get("target_field")
        condition = batch.get("condition")
        LF_field = batch.get("LF_field")

        # compute the loss
        loss = self._comp_loss(
            target_field=target_field, condition=condition, LF_field=LF_field
        )

        # log the loss
        self._log_losses(loss, stage="val")

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

    def _get_operator_config(self):
        """get the operator config"""
        operator_config = self.flow_config.get("operator")
        # check required keys in flow config
        required_keys = ["field", "method"]
        required_sub_keys = [
            "hidden_channels",
            "lifting_channel_ratio",
            "projection_channel_ratio",
            "modes",
            "n_layers",
        ]

        check_keys(operator_config, required_keys)
        check_keys(operator_config["field"], required_sub_keys)

        # add field details
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

    def _sample_noise_measure(self, batch_size: int, field_domain: torch.Tensor):
        """sample noise measure"""
        # prepare flattened domain
        domain_eval = field_domain.flatten(start_dim=2).transpose(1, 2).squeeze(0)

        # field shape
        field_channels = deep_get(self.shape_dict, ["field", "channels"])
        field_dims = field_domain.shape[2:]

        # sample noise
        noise_samples = self.noise.sample(
            domain=domain_eval,
            batch_size=batch_size,
            field_channels=field_channels,
            field_dims=field_dims,
        )
        check_tensor_blowup(noise_samples, name="noise samples")
        assert noise_samples.shape == (batch_size, field_channels, *field_dims)

        return noise_samples

    def _sample_prior_measure(
        self, batch_size: int, LF_field: torch.Tensor, field_domain: torch.Tensor
    ):
        """sample the base measure"""
        # prepare flattened domain
        domain_eval = field_domain.flatten(start_dim=2).transpose(1, 2).squeeze(0)

        # field shape
        field_channels = deep_get(self.shape_dict, ["field", "channels"])
        field_dims = field_domain.shape[2:]

        # sample prior
        noise_samples = self.prior.sample(
            domain=domain_eval,
            batch_size=batch_size,
            field_channels=field_channels,
            field_dims=field_dims,
        )
        check_tensor_blowup(noise_samples, name="noise samples")
        assert noise_samples.shape == (batch_size, field_channels, *field_dims)

        # add bias
        if self.floral:
            # prior_samples = LF_field + noise_samples
            prior_samples = noise_samples
        else:
            prior_samples = noise_samples

        return prior_samples

    def _sample_conditional_flow(
        self,
        target_field: torch.Tensor,
        prior: torch.Tensor,
        t: torch.Tensor,
        field_domain: torch.Tensor,
    ):
        """sample the conditional flow"""
        # check shape
        assert (
            prior.shape == target_field.shape
        ), "prior sample shape not same as field shape"
        # extract shape
        batch_size = target_field.shape[0]
        field_ndim = deep_get(self.shape_dict, ["field", "ndim"])
        # reshape t
        t_expand = t.view(batch_size, 1, *([1] * field_ndim))
        # sample noise (draw from noise measure)
        noise_scale = torch.mean(
            (target_field - prior) ** 2,
            dim=list(range(1, target_field.ndim)),
            keepdim=True,
        )
        noise = self._sample_noise_measure(
            batch_size=batch_size, field_domain=field_domain
        )
        noise = self.sig_min * noise_scale * noise
        # sample from conditional probability path
        psi = (t_expand * target_field + (1.0 - t_expand) * prior) + noise

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

        # compute the conditional flow
        psi_prime = target_field - prior

        assert (
            psi_prime.shape == target_field.shape
        ), "incorrect conditional flow derivative"

        return psi_prime

    def _comp_loss(
        self,
        target_field: torch.Tensor,
        condition: torch.Tensor,
        LF_field: torch.Tensor,
    ):
        """compute the flow-matching loss"""
        # slice
        target_field = target_field[self.slice_op]
        condition = condition[self.slice_op]
        LF_field = LF_field[self.slice_op]
        field_domain = self.field_domain[self.slice_op]

        # extract shape
        batch_size, field_channels, *field_dims = target_field.shape
        # sample time from U[0, 1]
        t = torch.rand(batch_size, 1, device=self.device)
        # sample prior measure
        prior = self._sample_prior_measure(
            batch_size=batch_size, LF_field=LF_field, field_domain=field_domain
        )
        # sample the conditional flow
        psi = self._sample_conditional_flow(
            target_field=target_field, prior=prior, t=t, field_domain=field_domain
        )
        # compute conditional flow derivative
        psi_prime = self._comp_conditional_flow_derivative(
            target_field=target_field, prior=prior
        )
        # compute the model vector field
        vt = self.vector_field(
            psi=psi,
            condition=condition,
            LF_field=LF_field,
            field_domain=self.field_domain,
            t=t,
        )
        assert vt.shape == psi_prime.shape, "incorrect target and model vector field"
        # Compute without time weighting
        # loss = torch.mean((vt - psi_prime)**2)
        # Compute time weighted loss
        w_t = 1.0 + 2.0 * t**2
        loss_raw = ((vt - psi_prime) ** 2).mean(dim=list(range(1, vt.ndim)))
        loss = (w_t.squeeze() * loss_raw).mean()
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

    def on_after_backward(self):
        """log gradient norm for training stability"""
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf"))
        self.log("grad_norm", grad_norm, on_step=True, on_epoch=False)

    def load_state_dict(self, state_dict, strict=True):
        # drop PyTorch-internal metadata if present
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)

    def _wrapper(
        self,
        field: torch.Tensor,
        condition: torch.Tensor,
        LF_field: torch.Tensor,
        t: torch.Tensor,
    ):
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
        LF_field_eval = LF_field.reshape(
            batch_size * n_gen, field_channels, *field_dims
        )
        t_eval = (
            torch.ones(batch_size * n_gen, 1, device=self.device) * t
        )  # (batch_size, 1)
        with torch.no_grad():
            vt = self.vector_field(
                psi=field_eval,
                condition=condition_eval,
                LF_field=LF_field_eval,
                field_domain=self.field_domain,
                t=t_eval,
            )
        return vt.view(batch_size, n_gen, field_channels, *field_dims)

    @torch.no_grad()
    def integrate_flow(
        self,
        condition: torch.Tensor,
        LF_field: torch.Tensor,
        generate_config: dict,
    ):
        """integrate the flow on the full domain"""
        # genernate config
        n_steps = generate_config.get("n_steps")
        n_gen = generate_config.get("n_gen")
        atol = generate_config.get("atol")
        rtol = generate_config.get("rtol")
        method = generate_config.get("method")

        # move to device
        condition = condition.to(self.device)
        LF_field = LF_field.to(self.device)

        # reshape to (B, n_gen, *dims)
        batch_size = len(condition)
        condition = condition.unsqueeze(1).repeat(
            1, n_gen, *([1] * (condition.ndim - 1))
        )
        LF_field = LF_field.unsqueeze(1).repeat(1, n_gen, *([1] * (LF_field.ndim - 1)))

        # interation times
        t = torch.linspace(0.0, 1.0, n_steps).to(self.device)

        # sample prior
        prior = self._sample_prior_measure(
            batch_size=batch_size * n_gen,
            LF_field=LF_field.view(-1, *LF_field.shape[2:]),
            field_domain=self.field_domain,
        )
        prior = prior.view(batch_size, n_gen, *prior.shape[1:]).to(self.device)

        # rhs
        def _rhs(t, xt):
            vt = self._wrapper(
                field=xt,
                condition=condition,
                LF_field=LF_field,
                t=t,
            )

            assert (
                vt.shape == xt.shape
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

        # check shape
        channels = deep_get(self.shape_dict, ["field", "channels"])
        dims = deep_get(self.shape_dict, ["field", "dims"])
        assert x1.shape == (batch_size, n_gen, channels, *dims)

        return x1
