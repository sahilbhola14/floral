import torch
import lightning as L
from contextlib import nullcontext
from torch.amp import autocast
from .utils_IO import printer, check_tensor_blowup
from tqdm import tqdm


class Inference:
    def __init__(
        self,
        model: L.LightningModule,
        val_set: torch.utils.data.TensorDataset,
        statistics: dict,
        job_name: str,
        floral: bool,
        generate_config: dict,
    ):
        # set model to eval mode
        self.model = model
        self.model.eval()
        # device
        self.device = self.model.device
        # enable inference optimizations
        torch.backends.cudnn.benchmark = True
        # set attributes
        self.statistics = statistics
        self.job_name = job_name
        self.floral = floral
        self.generate_config = generate_config
        # extract generate config
        self.minibatch_size = self.generate_config.get("minibatch_size", 10)
        # number of time steps
        self.nT = self.generate_config.get("nT", 10)
        # number of samples to generate per condition (for UQ)
        self.n_gen = self.generate_config.get("n_gen", 10)
        # pre-extract normalization factors
        self.target_field_mean = self.statistics["target_field"]["mean"]
        self.target_field_std = self.statistics["target_field"]["std"]
        # prepare inference input dict
        self.inference_input_dict = self._get_inference_input_dict(val_set)

    def _get_inference_input_dict(self, val_set: torch.utils.data.TensorDataset):
        """prepare the input(s) for the inference"""
        # check val set size
        assert (
            len(val_set.tensors) == 3
        ), "expected: (target_field, condition, LF_field)"
        # extract
        target_field, condition, LF_field = val_set.tensors
        # get shape
        field_channels = self.model.shape_dict["target_field"]["channels"]
        field_dims = self.model.shape_dict["target_field"]["dims"]
        # compute HF_field
        denormal_target_field = (
            target_field * self.target_field_std + self.target_field_mean
        )
        if self.floral:
            # if floral is True, target field is the residual
            HF_field = denormal_target_field + LF_field
        else:
            HF_field = denormal_target_field

        # domains
        field_domain = self.model.target_field_domain
        condition_domain = self.model.condition_domain

        # build the dict
        inference_input_dict = {
            "target_field": target_field,
            "condition": condition,
            "LF_field": LF_field,
            "HF_field": HF_field,
            "n_samples": len(target_field),
            "field_channels": field_channels,
            "field_dims": field_dims,
            "field_domain": field_domain,
            "condition_domain": condition_domain,
        }
        return inference_input_dict

    @torch.no_grad()
    def __call__(self):
        """perform the inference task"""
        # prepare the result dict
        n_samples = self.inference_input_dict.get("n_samples")
        field_channels = self.inference_input_dict.get("field_channels")
        field_dims = self.inference_input_dict.get("field_dims")
        result_dict = {
            "target_field": self.inference_input_dict.get("target_field"),
            "condition": self.inference_input_dict.get("condition"),
            "field_domain": self.inference_input_dict.get("field_domain"),
            "condition_domain": self.inference_input_dict.get("condition_domain"),
            "LF_field": self.inference_input_dict.get("LF_field"),
            "HF_field": self.inference_input_dict.get("HF_field"),
            "HF_field_prediction": torch.empty(
                n_samples, self.n_gen, field_channels, *field_dims, device="cpu"
            ),
        }
        # use amp
        # use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, "autocast")
        use_amp = False
        printer(f"Using automatic mixed-precision: {use_amp}")
        autocast_context = autocast("cuda") if use_amp else nullcontext()

        # create batches
        n_batches = (n_samples + self.minibatch_size - 1) // self.minibatch_size

        # iterate over batches
        pbar = tqdm(
            range(n_batches),
            desc="Inference",
            ncols=150,
            leave=True,
        )

        # get all the conditions (normalized for model input)
        condition = result_dict.get("condition").to(self.device)
        # get all the LF_field (unnormalized)
        LF_field = result_dict.get("LF_field")

        for ii in pbar:
            low_idx = ii * self.minibatch_size
            high_idx = (ii + 1) * self.minibatch_size
            with autocast_context:
                # get the condition batch
                condition_batch = condition[low_idx:high_idx].to(self.device)
                # get the LF_field batch
                LF_field_batch = LF_field[low_idx:high_idx]
                # prediction
                if self.floral:
                    # normalized prediction
                    residual_prediction_batch = self._get_prediction(
                        condition=condition_batch,
                    ).view(-1, field_channels, *field_dims)
                    # denormalize
                    residual_prediction_batch = (
                        residual_prediction_batch * self.target_field_std
                        + self.target_field_mean
                    )
                    residual_prediction_batch = residual_prediction_batch.view(
                        -1, self.n_gen, field_channels, *field_dims
                    )
                    # add LF_field to get the HF_field prediction
                    HF_field_prediction_batch = (
                        residual_prediction_batch + LF_field_batch.unsqueeze(1)
                    )
                else:
                    raise NotImplementedError

            # store the result
            result_dict["HF_field_prediction"][
                low_idx:high_idx
            ] = HF_field_prediction_batch

        # ensure all async transfers are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # check tensor blow up
        check_tensor_blowup(result_dict["HF_field_prediction"])

        # save the results to a file
        save_path = f"{self.job_name}_results{'_floral' if self.floral else ''}.pt"
        printer(f"Saving results to {save_path}")
        torch.save(result_dict, save_path)

    def _get_prediction(self, condition: torch.Tensor):
        """integrate the ODE"""
        prediction = self.model.integrate_flow(
            condition=condition,
            **self.generate_config,
        )
        return prediction.cpu()
