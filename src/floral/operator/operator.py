import sys
import time
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

# from floral.archs import get_vector_field_operator
from floral.archs import build_fno

# from floral.gp import get_gp_prior
from collections.abc import Mapping
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field


@dataclass
class OperatorConfig:
    """Default config for Operator with override support."""

    # config defaults
    config: dict = field(
        default_factory=lambda: {
            "floral": False,
            "flow": {
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


class Operator(L.LightningModule):
    """Multi-fidelity Neural Operator"""

    def __init__(
        self,
        config: dict,
        hp_config: wandb.sdk.wandb_config.Config | dict,
        domain_dict: dict[str, torch.Tensor],
        shape_dict: dict[str, int],
    ):
        super(Operator, self).__init__()
        # merge defaults with user-provided runtime config/hp_config
        config_defaults = OperatorConfig()
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

        # extract operator config
        self.flow_config = self.config.get("flow")
        # check train resolution
        self._check_train_resolution()
        # get slice operator
        self.slice_op = self._get_slice_operator()
        # build operator config
        operator_config = self._get_operator_config()
        self.operator_method = operator_config.get("method")
        # build the operator modules for the vector field
        self.operator = build_fno(operator_config=operator_config)

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

    def _log_losses(self, loss_dict: dict, stage: str):
        """log all losses with stage prefix"""
        log_kwargs = {"on_step": False, "on_epoch": True, "sync_dist": True}
        self.log(f"{stage}_loss", loss_dict["loss"], prog_bar=True, **log_kwargs)
        self.log(f"{stage}_loss_unweighted", loss_dict["loss_unweighted"], **log_kwargs)
        self.log(f"{stage}_rel_l2", loss_dict["rel_l2"], **log_kwargs)

    def training_step(self, batch, batch_idx):
        """training step"""
        # extract the batch
        required_keys = ["target_field", "condition", "LF_field"]
        check_keys(batch, required_keys)
        target_field = batch.get("target_field")
        condition = batch.get("condition")
        LF_field = batch.get("LF_field")

        # compute the loss
        loss_dict = self._comp_loss(
            target_field=target_field, condition=condition, LF_field=LF_field
        )

        # log the loss
        self._log_losses(loss_dict, stage="train")
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        """validation step"""
        # extract the batch
        required_keys = ["target_field", "condition", "LF_field"]
        check_keys(batch, required_keys)
        target_field = batch.get("target_field")
        condition = batch.get("condition")
        LF_field = batch.get("LF_field")

        # compute the loss
        loss_dict = self._comp_loss(
            target_field=target_field, condition=condition, LF_field=LF_field
        )

        # log the loss
        self._log_losses(loss_dict, stage="val")
        return loss_dict["loss"]

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

        # inputs and the outputs: condition -> field
        operator_config["in_channels"] = deep_get(
            operator_config, ["condition", "channels"]
        )
        operator_config["out_channels"] = deep_get(
            operator_config, ["field", "channels"]
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

    def _forward_operator(
        self,
        condition: torch.Tensor,
    ):
        if self.operator_method == "FiLMFNO":
            return self.operator(x=condition, cond=condition)
        elif self.operator_method == "FNO":
            return self.operator(
                condition,
            )
        else:
            raise ValueError(f"{self.operator_method} not valid")

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
        # field_domain = self.field_domain[self.slice_op]

        # compute the prediction
        prediction = self._forward_operator(condition=condition)
        assert prediction.shape == target_field.shape, "incorrect prediction shape"
        spatial_dims = list(range(1, prediction.ndim))

        # per-sample squared errors
        err_sq = ((prediction - target_field) ** 2).mean(dim=spatial_dims)  # (B,)

        # training objective
        loss = err_sq.mean()

        # unweighted MSE — shows raw fit quality without time emphasis
        loss_unweighted = err_sq.mean()

        # relative L2 — scale-invariant, comparable across runs
        target_norm_sq = (target_field**2).mean(dim=spatial_dims).mean()
        rel_l2 = loss_unweighted / (target_norm_sq + 1e-8)

        return {
            "loss": loss,
            "loss_unweighted": loss_unweighted,
            "rel_l2": rel_l2,
        }

    def on_fit_start(self):
        """log static run info once at the start of training"""
        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        summary = {
            "n_params_total": n_total,
            "n_params_trainable": n_trainable,
            "n_train_samples": (
                self.trainer.datamodule.n_train if self.trainer.datamodule else None
            ),
        }
        if self.logger and hasattr(self.logger, "experiment"):
            self.logger.experiment.summary.update(summary)
        self._fit_start_time = time.time()
        self._best_val_loss = float("inf")
        self._best_epoch = 0

    def on_validation_epoch_end(self):
        """track running best val loss"""
        val_loss = self.trainer.callback_metrics.get("val_loss")
        if val_loss is not None and val_loss.item() < self._best_val_loss:
            self._best_val_loss = val_loss.item()
            self._best_epoch = self.current_epoch

    def on_fit_end(self):
        """log summary metrics at end of training"""
        train_time_s = time.time() - self._fit_start_time
        final_val_loss = self.trainer.callback_metrics.get("val_loss")
        summary = {
            "train_time_s": train_time_s,
            "time_per_epoch_s": train_time_s / max(self.current_epoch, 1),
            "best_val_loss": self._best_val_loss,
            "best_epoch": self._best_epoch,
            "final_val_loss": (
                final_val_loss.item() if final_val_loss is not None else None
            ),
            "n_epochs_trained": self.current_epoch,
        }
        if torch.cuda.is_available():
            summary["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        if self.logger and hasattr(self.logger, "experiment"):
            self.logger.experiment.summary.update(summary)

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
        # log the gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf"))
        self.log("grad_norm", grad_norm, on_step=True, on_epoch=False)

    def load_state_dict(self, state_dict, strict=True):
        # drop PyTorch-internal metadata if present
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)
