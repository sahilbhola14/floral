# src/floral/flow/flow.py
# import math
import sys
import lightning as L
import matplotlib.pyplot as plt
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
            # conver value to tensors
            v_tensor = torch.Tensor(v)
            self.register_buffer(f"{k}_domain", v_tensor, persistent=True)
        # floral
        self.floral = self.config.get("floral", True)
        # extract flow config
        flow_config = self.config["flow"]
        self.sig_min = flow_config.get("sig_min", 1e-5)
        # build operator config
        operator_config = self._get_operator_config(flow_config=flow_config)
        # build the operator modules for the vector field
        self.vector_field = get_vector_field_operator(
            operator_config=operator_config, floral=self.floral
        )
        # build the prior (eval mode implicit)
        self.prior_config = flow_config["prior"]
        if self.floral:
            self.prior = get_gp_prior(
                lengthscale=self.prior_config.get("lengthscale", 1e-3),
                outputscale=self.prior_config.get("outputscale", 1.0),
                confidence=self.prior_config.get("confidence", 0.0),
            )
        # noise
        self.noise = get_gp_prior(
            lengthscale=self.prior_config.get("lengthscale", 1e-3),
            outputscale=self.prior_config.get("outputscale", 1.0),
            confidence=0.0,  # no bias should be added to the noise
        )

        self.debug_plot = False

    def training_step(self, batch, batch_idx):
        """training step"""
        assert len(batch) == 3, "expected: (target_field, condition, LF_field)"
        target_field, condition, LF_field = batch
        # compute the loss
        loss = self._comp_loss(
            target_field=target_field, condition=condition, LF_field=LF_field
        )
        # log the training loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step"""
        assert len(batch) == 3, "expected: (target_field, condition, LF_field)"
        target_field, condition, LF_field = batch
        # compute the loss
        loss = self._comp_loss(
            target_field=target_field, condition=condition, LF_field=LF_field
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

    def _get_operator_config(self, flow_config):
        """get the operator config"""
        operator_config = flow_config.get("operator")
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

    def _sample_noise_measure(self, batch_size: int):
        """sample noise measure"""
        # prepare flattened domain
        domain_eval = self.field_domain.flatten(start_dim=2).transpose(1, 2).squeeze(0)

        # field shape
        field_channels = deep_get(self.shape_dict, ["field", "channels"])
        field_dims = deep_get(self.shape_dict, ["field", "dims"])
        # sample prior
        noise_samples = self.noise.sample(
            domain=domain_eval,
            batch_size=batch_size,
            field_channels=field_channels,
            field_dims=field_dims,
        )

        check_tensor_blowup(noise_samples, name="noise samples")

        assert noise_samples.shape == (batch_size, field_channels, *field_dims)

        return noise_samples

    def _sample_prior_measure(self, batch_size: int, LF_field: torch.Tensor):
        """sample the base measure"""
        noise_samples = self._sample_noise_measure(batch_size)

        if self.floral:
            # prior_samples = LF_field + noise_samples
            prior_samples = noise_samples
        else:
            prior_samples = noise_samples

        if self.debug_plot:
            if self.shape_dict["field"]["ndim"] == 2:
                fig, axs = plt.subplots(1, 3, figsize=(6, 2), layout="compressed")
                imgs = [None] * 3
                imgs[0] = axs[0].imshow(LF_field[0][0].detach().cpu().numpy())
                axs[0].set_title("Low-fidelity")
                imgs[1] = axs[1].imshow(noise_samples[0][0].detach().cpu().numpy())
                axs[1].set_title("Prior Noise")
                imgs[2] = axs[2].imshow(prior_samples[0][0].detach().cpu().numpy())
                axs[2].set_title("x0")
                for ii, ax in enumerate(axs):
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.colorbar(imgs[ii], ax=axs[ii], fraction=0.45, pad=0.1)
                plt.savefig("prior_floral.png" if self.floral else "prior_flora.png")
            else:
                fig, axs = plt.subplots(
                    1, 3, figsize=(6, 2), layout="compressed", sharex=True, sharey=True
                )
                axs[0].plot(LF_field[0][0].detach().cpu().numpy())
                axs[0].set_title("Low-fidelity")
                axs[1].plot(noise_samples[0][0].detach().cpu().numpy())
                axs[1].set_title("Prior Noise")
                axs[2].plot(prior_samples[0][0].detach().cpu().numpy())
                axs[2].set_title("x0")
                for ii, ax in enumerate(axs):
                    ax.set_xticks([])
                    # ax.set_yticks([])
                plt.savefig("prior_floral.png" if self.floral else "prior_flora.png")

        return prior_samples

    def _sample_conditional_flow(
        self, target_field: torch.Tensor, prior: torch.Tensor, t: torch.Tensor
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
        # sample noise (fresh draw from prior measure)
        noise = self._sample_noise_measure(batch_size)
        # sample conditional flow
        psi = (
            t_expand * target_field + (1.0 - t_expand) * prior
        ) + self.sig_min * noise

        if self.debug_plot:
            t_test = (
                torch.linspace(0, 1, 10)
                .view(10, 1, *([1] * field_ndim))
                .to(self.device)
            )
            x0_test = prior[0].unsqueeze(0)
            x1_test = target_field[0].unsqueeze(0)

            psi_test = t_test * x1_test + (1.0 - t_test) * x0_test
            noise_test = self._sample_noise_measure(10)
            samp_test = psi_test + self.sig_min * noise_test

            fig, axs = plt.subplots(
                5, 2, figsize=(4, 10), sharex=True, sharey=True, layout="compressed"
            )
            if self.shape_dict["field"]["ndim"] == 2:
                for ii, ax in enumerate(axs.flatten()):
                    im = ax.imshow(samp_test[ii][0].detach().cpu().numpy())
                    # im = ax.imshow(psi_test[ii][0].detach().cpu().numpy())
                    fig.colorbar(im, ax=ax, fraction=0.45, pad=0.1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                plt.savefig("psi_floral.png" if self.floral else "psi_flora.png")
            else:
                for ii, ax in enumerate(axs.flatten()):
                    ax.plot(samp_test[ii][0].detach().cpu().numpy())
                    ax.set_xticks([])
                plt.savefig("psi_floral.png" if self.floral else "psi_flora.png")

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

        if self.debug_plot:
            if self.shape_dict["field"]["ndim"] == 2:
                fig, axs = plt.subplots(1, 3, layout="compressed", figsize=(6, 2))
                imgs = [None] * 3
                imgs[0] = axs[0].imshow(target_field[0][0].detach().cpu().numpy())
                axs[0].set_title("High-fidelity")
                imgs[1] = axs[1].imshow(prior[0][0].detach().cpu().numpy())
                axs[1].set_title("x0")
                imgs[2] = axs[2].imshow(psi_prime[0][0].detach().cpu().numpy())
                axs[2].set_title("psi_prime")
                for ii, im in enumerate(imgs):
                    fig.colorbar(im, ax=axs[ii], fraction=0.4, pad=0.1)
                    axs[ii].set_xticks([])
                    axs[ii].set_yticks([])
                plt.savefig(
                    "psiprime_floral.png" if self.floral else "psiprime_flora.png"
                )
            else:
                fig, axs = plt.subplots(
                    1, 3, layout="compressed", figsize=(6, 2), sharey=True, sharex=True
                )
                axs[0].plot(target_field[0][0].detach().cpu().numpy())
                axs[0].set_title("High-fidelity")
                axs[1].plot(prior[0][0].detach().cpu().numpy())
                axs[1].set_title("x0")
                axs[2].plot(psi_prime[0][0].detach().cpu().numpy())
                axs[2].set_title("psi_prime")
                for ii in range(len(axs.flatten())):
                    axs[ii].set_xticks([])
                    # axs[ii].set_yticks([])
                plt.savefig(
                    "psiprime_floral.png" if self.floral else "psiprime_flora.png"
                )

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
        # extract shape
        batch_size, field_channels, *field_dims = target_field.shape
        # sample time from U[0, 1]
        t = torch.rand(batch_size, 1, device=self.device)
        # t = torch.cos(torch.rand(batch_size, 1, device=self.device)*math.pi/2.0)
        # sample prior measure
        prior = self._sample_prior_measure(batch_size=batch_size, LF_field=LF_field)
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
        if self.debug_plot:
            sys.exit()
        # regularization to pay attention where vector field magnitude is large
        # reg = ((psi_prime)**2).mean(dim=list(range(1, vt.ndim)))
        # reg_loss = loss + reg.mean()
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
        # create LF_field_batch
        LF_field_batch = (
            LF_field.unsqueeze(1)
            .expand(-1, n_gen, field_channels, *field_dims)
            .to(self.device)
        )
        LF_field_reshaped = LF_field_batch.reshape(-1, field_channels, *field_dims)
        # sample prior
        prior = self._sample_prior_measure(
            batch_size=batch_size * n_gen, LF_field=LF_field_reshaped
        ).view(batch_size, n_gen, field_channels, *field_dims)

        prior = prior.to(self.device)

        # interation times
        t = torch.linspace(0, 1, nT).to(self.device)

        # RHS
        def _rhs(t, field):
            vt = self._wrapper(
                field=field,
                condition=condition_batch,
                LF_field=LF_field_batch,
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
