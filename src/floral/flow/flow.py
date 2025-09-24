# src/floral/flow/flow.py
"""
Flow-matching operator class.
"""
import wandb
import math
import torch
from abc import ABC, abstractmethod
from torchdiffeq import odeint
from floral.utils import printer


class Flow(ABC):
    """Base class for flow models.
    Attributes:
        hp_config (wand.sdk.wandb_config.Config | dict): hyperparameter dict
        domains (dict): Dictionary for the field and condition domain
        (flattened, C-type).
        flow_config (dict): Dictionary for the flow config
    """

    def __init__(
        self,
        hp_config: wandb.sdk.wandb_config.Config | dict,
        domains: dict,
        flow_config: dict,
    ):
        """Initialize the flow model."""
        # hyper parameters
        self.hp_config = dict(hp_config)
        # get the domain info for the field and condition
        self.field_domain, self.condition_domain = self._get_domain_info(domains)
        # flow config
        self.flow_config = dict(flow_config)
        self._set_flow_config()
        # stepper config
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

    def _get_domain_info(self, domains: dict):
        """extract the domain dict
        Args:
            domains (dict): Dictionary with field and condition domain info.
        Returns:
            field_domain (torch.Tensor): Flattened field domain.
            For example if the field if of shape (B, num_channels, *dim),
            then field_domain is of shape (np.prod(dim), len(dim))
            condition_domain (torch.Tensor): Flattened condition domain.
            For example if the condition if of shape (B, num_channels, *dim),
            then condition_domain is of shape (np.prod(dim), len(dim))
        """
        assert (
            "field" in domains.keys() and "condition" in domains.keys()
        ), "field/condition domain key unavailable. Check domains dict."
        field_domain = domains.get("field", None)
        condition_domain = domains.get("condition", None)
        assert isinstance(
            field_domain, torch.Tensor
        ), "Expected torch.Tensor, got {type(field_domain).__name__}"
        assert isinstance(
            condition_domain, torch.Tensor
        ), "Expected torch.Tensor, got {type(condition_domain).__name__}"

        return field_domain, condition_domain

    def _set_flow_config(self):
        """set the flow config"""
        # check for the required keys
        required_keys = [
            "sig_min",
            "field_sub_sample_factor",
            "condition_sub_sample_factor",
            "time_embed_freq",
        ]
        missing_keys = [k for k in required_keys if k not in self.flow_config]
        if missing_keys:
            raise KeyError(f"Missing keys in flow config: {missing_keys}")

        # sigma min
        self.sig_min = self.flow_config.get("sig_min", 1e-5)
        assert self.sig_min > 0, "Need positive sigma min."

        # sub - sampling factor
        def _check_factor(factors, domain):
            resolution, dim = domain.shape
            assert len(factors) > 0, "atleast one sub-sampling factor must be provided"
            assert all(
                [isinstance(ii, int) for ii in factors]
            ), "sub-sample factors must be a list of integers"
            if dim == 1:
                assert all(
                    [(resolution % ii == 0) for ii in factors]
                ), "sub-sampling must perfectly divide the resolution"
            elif dim == 2:
                raise NotImplementedError
                xy_res = int(math.sqrt(resolution))
                assert (
                    xy_res**2 == resolution
                ), "currently, only square images are supported"
                assert all(
                    [(xy_res % ii == 0) for ii in factors]
                ), "sub-sampling must perfectly divide the resolution"
            else:
                raise ValueError("3D field not currently implemented")

        self.field_sub_sample_factor = self.flow_config.get(
            "field_sub_sample_factor", [1]
        )
        _check_factor(self.field_sub_sample_factor, self.field_domain)

        self.condition_sub_sample_factor = self.flow_config.get(
            "condition_sub_sample_factor", [1]
        )
        _check_factor(self.condition_sub_sample_factor, self.condition_domain)

        # time_embed_freq
        self.time_embed_freq = self.flow_config.get("time_embed_freq", 4)
        assert (
            isinstance(self.time_embed_freq, int) and self.time_embed_freq > 0
        ), "time embedding freq must be a positive integer."
        self.time_embed_dim = 2 * self.time_embed_freq

    def _sub_sample_condition(self, condition: torch.Tensor):
        """sub sample the condition
        Args:
            condition (torch.Tensor):
                Input (full) condition tensor of shape
                (B, condition_ch_in, *condition_grid_full)
        Returns:
            condition_slices (torch.Tensor):
                Sliced condition tensor of shape
                (B, condition_ch_in, *condition_grid), where conditon_grid size
                is dependent on number of spatial sensors placed.
        """
        B, condition_ch, *condition_grid = condition.shape
        # sub sample factor
        idx = int(torch.randint(0, len(self.condition_sub_sample_factor), (1,)))
        ss_factor = self.condition_sub_sample_factor[idx]
        # slice idx
        slices_idx = (slice(None), slice(None)) + (slice(None, None, ss_factor),) * (
            condition.ndim - 2
        )
        # reshape domain
        condition_domain_reshaped = self.condition_domain.T.view(
            -1, *condition_grid
        ).unsqueeze(0)
        # slices
        condition_slices = condition[slices_idx]
        condition_domain_slices = condition_domain_reshaped[slices_idx]

        return condition_slices, condition_domain_slices

    def _sub_sample_field(self, field: torch.Tensor):
        """sub sample the field"""
        B, field_ch, *field_grid = field.shape
        # sub sample factor
        idx = int(torch.randint(0, len(self.field_sub_sample_factor), (1,)))
        ss_factor = self.field_sub_sample_factor[idx]
        # slice idx
        slices_idx = (slice(None), slice(None)) + (slice(None, None, ss_factor),) * (
            field.ndim - 2
        )
        # reshape domain
        field_domain_reshaped = self.field_domain.T.view(-1, *field_grid).unsqueeze(0)
        # slices
        field_samples = field[slices_idx]
        field_domain_samples = field_domain_reshaped[slices_idx]
        return field_samples, field_domain_samples

    def _compute_conditional_flow(
        self,
        prior: torch.Tensor,
        field: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ):
        """compute the conditional flow"""
        assert prior.shape == field.shape, "prior sample shape not same as field shape"
        B, field_ch, *field_grid = field.shape
        t_expand = t.view(B, 1, *([1] * len(field_grid)))
        return (t_expand * field + (1.0 - t_expand) * prior) + self.sig_min * noise

    def _compute_conditional_flow_derivative(
        self, prior: torch.Tensor, field: torch.Tensor
    ):
        """compute the derivative (time) of the conditional flow"""
        assert prior.shape == field.shape, "prior sample shape not same as field shape"
        return field - prior

    def _compute_loss(self, field: torch.Tensor, condition: torch.Tensor):
        """compute the vector field regression loss
        Args:
            field (torch.Tensor): samples from conditional measure
            :math:`\\nu_1(\\cdot\\vert c)` of shape (B, ch_f, *dim_f)
            condition (torch.Tensor): samples from the condition measure
            :math:`\\nu_c(c)` of shape (B, ch_c, *dim_c)
        """
        # sample time from U[0, 1]
        t = torch.rand(len(field), 1, device=self.device)
        # spatially sub-sample the condition
        condition_samples, condition_domain_samples = self._sub_sample_condition(
            condition=condition
        )
        # spatially sub-sample the field
        field_samples, field_domain_samples = self._sub_sample_field(field=field)
        # Sample base measure (prior)
        batch_size, field_ch, *field_grid = field_samples.shape
        prior_samples = self._sample_base_measure(
            field_domain=field_domain_samples,
            field_grid=field_grid,
            field_ch=field_ch,
            n_samples=batch_size,
        )
        # noise samples (fresh draw from base measure)
        noise_samples = self._sample_base_measure(
            field_domain=field_domain_samples,
            field_grid=field_grid,
            field_ch=field_ch,
            n_samples=batch_size,
        )
        # Compute the conditional flow
        psi_samples = self._compute_conditional_flow(
            prior=prior_samples, field=field_samples, t=t, noise=noise_samples
        )
        # Compute the conditional flow derivative
        psi_prime = self._compute_conditional_flow_derivative(
            prior=prior_samples, field=field_samples
        )
        # Compute the vector field
        vt = self._evaluate_vector_field(
            psi=psi_samples,
            condition=condition_samples,
            field_domain=field_domain_samples,
            condition_domain=condition_domain_samples,
            t=t,
        )
        assert (
            vt.shape == psi_prime.shape
        ), "invalid target and model vector field shape"
        # Compute the loss
        loss = torch.mean((vt - psi_prime) ** 2)
        return loss

    def _time_embedding(
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

    def _evaluate_vector_field(
        self,
        psi: torch.Tensor,
        condition: torch.Tensor,
        field_domain: torch.Tensor,
        condition_domain: torch.Tensor,
        t: torch.Tensor,
    ):
        """evalute the vector field of the flow
        Args:
            psi (torch.Tensor):
                Samples from the conditional probability path measure
                :math:`\\mu_\\tau(w\\vert a,z;\\theta)`
            condition (torch.Tensor):
                Samples from the input condition measure :math:`\\nu_a`
            field_domain (torch.Tensor):
                Domain for the field conditioning
            condition_domain (torch.Tensor):
                Domain for the condition conditioning
            t (torch.Tensor):
                Samples of the pseudo-time
        """
        # embedd the time (B, time_embed_dim)
        time_embed = self._time_embedding(t=t, n_freq=self.time_embed_freq)
        # embedd the condition (B, condition_ch_out, *condition_grid)
        condition_embed = self.condition_operator(
            condition=condition,
            condition_domain=condition_domain,
            time_embed=time_embed,
        )
        # embedd the field v_t(w, a)
        vt = self.field_operator(
            field=psi,
            field_domain=field_domain,
            condition_embed=condition_embed,
            time_embed=time_embed,
        )
        assert (
            vt.shape == psi.shape
        ), f"Invalid shape of the vector field. Expected {psi.shape}, got {vt.shape}"

        return vt

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

    def training_step(self, batch, batch_idx):
        """training step"""
        field, condition, _ = batch  # (target samples, condition, low fidelity)
        loss = self._compute_loss(field=field, condition=condition)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation step"""
        field, condition, _ = batch  # (target samples, condition, low fidelity)
        loss = self._compute_loss(field=field, condition=condition)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

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
    def _sample_base_measure(
        self,
        field_domain: torch.Tensor,
        field_grid: tuple | list,
        field_ch: int,
        n_samples: int,
    ):
        """sample the base measure"""
        pass

    def _sample_initial_condition(self, c: torch.Tensor, batch_size: int, n_gen: int):
        """get the initial condition for the flow"""
        raise NotImplementedError
        pass
