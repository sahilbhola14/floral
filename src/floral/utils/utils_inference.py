import math
import torch
import lightning as L
import gpytorch
from .utils_IO import printer, check_tensor_blowup
from tqdm import tqdm
from gpytorch.models import ExactGP
from contextlib import nullcontext
from torch.amp import autocast


class Inference:
    """Inference class for handling inference operations.
    Attributes:
        model (L.LightningModule):
            Best model for inference tasks
        val_set (torch.utils.data.TensorDataset):
            Tensor dataset (field, condition, LF_field)
        domains (dict):
            Domain information for the field and the condition.
            Contains the full domain for inference
        statistics (dict):
            statistics for the training data, used to denormalize the prediction
        job_name (str):
            job name identifier for saving
        floral (bool):
            True means model predicts the residual correction
        generate_config (dict):
            Configurations for the generative process
    """

    def __init__(
        self,
        model: L.LightningModule,
        val_set: torch.utils.data.TensorDataset,
        domains: dict,
        statistics: dict,
        job_name: str,
        floral: bool,
        generate_config: dict,
    ):
        # Initialize the model with the best_model_path
        self.model = model
        # Set the model to evaluation mode
        self.model.eval()
        # Device
        self.device = self.model.device
        # Enable inference optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes

        # Store the test configuration, statistics, job name, and number of samples
        self.statistics = statistics  # statistics of the data
        self.job_name = job_name  # name of the inference job
        self.floral = floral  # flag for multi-flow processing
        self.generate_config = generate_config  # configuration for generation settings

        # extract generate config
        # number of conditions to be evaluated at once
        self.minibatch_size = generate_config.get("minibatch_size", 10)
        # number of time steps
        self.nT = generate_config.get("nT", 10)  # Number of time steps
        # number of samples for the field per condition (for forward UQ)
        self.n_gen = generate_config.get("n_gen", 10)

        # pre-extract normalization factors
        self.field_mean = self.statistics["field"]["mean"].to(self.device)
        self.field_std = self.statistics["field"]["std"].to(self.device)

        # prepare inference input dict
        self.inference_input_dict = self._get_inference_input_dict(val_set, domains)

    def _get_inference_input_dict(
        self, val_set: torch.utils.data.TensorDataset, domains: dict
    ):
        """prepare the input(s) for the inference"""
        # get domain info
        field_domain, condition_domain = self._get_domain_info(domains)

        assert len(val_set.tensors) == 3, "expected (field, condition, LF_field)"
        field, condition, LF_field = val_set.tensors
        # check shapes
        field_B, field_ch_in, *field_grid = field.shape
        condition_B, condition_ch_in, *condition_grid = condition.shape
        LF_field_B, LF_field_ch_in, *LF_field_grid = LF_field.shape
        assert field_B == condition_B == LF_field_B, "incorrect number of samples"
        assert math.prod(field_grid) == len(
            field_domain
        ), "inconsistent field domain and field"
        assert math.prod(condition_grid) == len(
            condition_domain
        ), "inconsistent condition domain and condition"
        assert math.prod(LF_field_grid) == len(
            field_domain
        ), "inconsistent field domain and field"
        assert field_domain.shape[1] == len(field_grid), "invalid domain"
        assert condition_domain.shape[1] == len(condition_grid), "invalid domain"

        # reshape domain
        field_domain = field_domain.T.view(-1, *field_grid).unsqueeze(0)
        condition_domain = condition_domain.T.view(-1, *condition_grid).unsqueeze(0)

        # build dict
        inference_input_dict = {
            "field": field,
            "field_domain": field_domain,
            "condition": condition,
            "condition_domain": condition_domain,
            "LF_field": LF_field,
            "field_ch": field_ch_in,
            "field_grid": field_grid,
            "n_samples": len(field),
        }

        return inference_input_dict

    def _get_domain_info(self, domains: dict):
        """extract the domain dict
        Args:
            domains (dict):
                Dictionary with field and condition domain info.
        Returns:
            field_domain (torch.Tensor):
                Flattened field domain. For example if the field if of shape
                (B, num_channels, *dim), then field_domain is of shape
                (np.prod(dim), len(dim))
            condition_domain (torch.Tensor):
                Flattened condition domain. For example if the condition if of shape
                (B, num_channels, *dim), then condition_domain is of shape
                (np.prod(dim), len(dim))
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

    @torch.no_grad()
    def __call__(self):
        """Perform the inference task"""
        # pre-allocate tensors
        n_samples = self.inference_input_dict.get("n_samples")
        field_ch = self.inference_input_dict.get("field_ch")
        field_grid = self.inference_input_dict.get("field_grid")

        result_dict = {
            "field": self.inference_input_dict.get("field"),
            "field_domain": self.inference_input_dict.get("field_domain"),
            "condition": self.inference_input_dict.get("condition"),
            "condition_domain": self.inference_input_dict.get("condition_domain"),
            "LF_field": self.inference_input_dict.get("LF_field"),
            "field_prediction": torch.empty(
                n_samples, self.n_gen, field_ch, *field_grid, device="cpu"
            ),
            "residual_prediction": torch.empty(
                n_samples, self.n_gen, field_ch, *field_grid, device="cpu"
            ),
        }

        # use mixed precision if available for faster inference
        # use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, "autocast")
        # TODO: Make FNO 16-bit friendly
        use_amp = False  # FNO does not work out of the box with 16-bit
        printer(f"Using automatic mixed-precision: {use_amp}")
        autocast_context = autocast("cuda") if use_amp else nullcontext()

        n_batches = (n_samples // self.minibatch_size) + 1

        pbar = tqdm(
            range(n_batches),
            desc="Inference",
            ncols=150,
            leave=True,
        )

        for ii in pbar:
            # field (high-precision)
            field = result_dict.get("field")[
                ii * self.minibatch_size : (ii + 1) * self.minibatch_size
            ]
            field = field.unsqueeze(1).to(self.device)  # (B, 1, field_ch, *field_grid)
            # low-precision result
            LF_field = result_dict.get("LF_field")[
                ii * self.minibatch_size : (ii + 1) * self.minibatch_size
            ]
            LF_field = LF_field.unsqueeze(1).to(
                self.device
            )  # (B, 1, field_ch, *field_grid)
            with autocast_context:
                # get the condition mini-batch
                c_eval = result_dict.get("condition")[
                    ii * self.minibatch_size : (ii + 1) * self.minibatch_size
                ]
                c_eval = c_eval.to(self.device)

                # perform the prediction
                if self.floral:
                    pred_residual = self._get_prediction(c_eval=c_eval).view(
                        -1, field_ch, *field_grid
                    )
                    # Denormalize in-place
                    pred_residual = pred_residual * self.field_std + self.field_mean
                    pred_residual = pred_residual.view(
                        -1, self.n_gen, field_ch, *field_grid
                    )
                    # Get the high fidelity field prediction
                    pred_field = pred_residual + LF_field
                    pred_field = pred_field.to(
                        "cpu", non_blocking=True
                    )  # Move back to CPU
                    pred_residual = pred_residual.to("cpu", non_blocking=True)
                else:
                    pred_field = self._get_prediction(c_eval=c_eval).view(
                        -1, field_ch, *field_grid
                    )
                    # Denormalize in-place
                    pred_field = pred_field * self.field_std + self.field_mean
                    pred_field = pred_field.view(-1, self.n_gen, field_ch, *field_grid)
                    pred_field_cpu = pred_field.to("cpu", non_blocking=True)
                    # Compute the residual
                    pred_residual = (pred_field - LF_field).to("cpu", non_blocking=True)
                    pred_field = pred_field_cpu

            # store the fieeld
            result_dict["field_prediction"][
                ii * self.minibatch_size : (ii + 1) * self.minibatch_size
            ] = pred_field
            # store the residual
            result_dict["residual_prediction"][
                ii * self.minibatch_size : (ii + 1) * self.minibatch_size
            ] = pred_residual

        # Ensure all async transfers are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # check for nans
        check_tensor_blowup(result_dict["field_prediction"], "predicted field")
        check_tensor_blowup(result_dict["residual_prediction"], "predicted residual")

        # Save the results to a file
        save_path = f"{self.job_name}_results{'_floral' if self.floral else ''}.pt"
        printer(f"Saving results to {save_path}")
        torch.save(result_dict, save_path)

        pbar.close()

    def _get_prediction(self, c_eval: torch.Tensor):
        """Get the model prediction for a given conditon over the entire domain
        Args:
            c_eval (torch.Tensor): Input conditons to generate the solution
        Returns:
            prediction (torch.Tensor): Model prediction
        """
        # domains
        field_domain = self.inference_input_dict.get("field_domain").to(self.device)
        condition_domain = self.inference_input_dict.get("condition_domain").to(
            self.device
        )

        # get the prediction for c_eval conditions
        prediction = self.model.flow.interpolate(
            condition=c_eval,
            condition_domain=condition_domain,
            field_domain=field_domain,
            field_ch=self.inference_input_dict.get("field_ch"),
            field_grid=self.inference_input_dict.get("field_grid"),
            n_gen=self.n_gen,
            nT=self.nT,
            method=self.generate_config.get("method", "dopri5"),
            atol=self.generate_config.get("atol", 1e-4),
            rtol=self.generate_config.get("rtol", 1e-4),
        )

        return prediction


class InferenceGP:
    """Inference clas for Gaussian Process"""

    def __init__(
        self,
        model: ExactGP,
        test_config: dict,
        statistics: dict,
        job_name: str,
        floral: bool,
        device: torch.device,
    ):
        # Initialize the model with the best_model_path
        self.model = model
        # Set the model to evaluation mode
        self.model.eval()
        # Set the likelhood to evaluation mode
        self.model.likelihood.eval()
        # Store the test configuration, statistics, job name, and number of samples
        self.test_config = test_config  # Configuration for the inference task
        self.statistics = statistics  # Statistics of the data
        self.job_name = job_name  # Name of the inference job
        self.floral = floral  # Flag for multi-flow processing
        self.device = device

    @torch.no_grad()
    def __call__(self):
        """Perform the inference task"""
        field = {"LF_field": [], "HF_field": [], "Prediction": {"mean": [], "std": []}}
        residual = {"True": [], "Prediction": {"mean": [], "std": []}}
        # Input featurs
        in_features = self.test_config["in_features"].to(self.device)
        # Fields
        LF_field = self.test_config["LF_field"]
        HF_field = self.test_config["HF_field"]
        # Prediction
        batch_size = 2000
        pred_mean = []
        pred_std = []
        with gpytorch.settings.fast_pred_var():
            for ii in range(0, in_features.size(0), batch_size):
                in_features_batch = in_features[ii : ii + batch_size]
                pred = self.model.likelihood(self.model(in_features_batch))
                pred_mean.append(pred.mean)
                pred_std.append(pred.stddev)
        # Concatenate the predictions
        pred_mean = torch.cat(pred_mean, dim=0)
        pred_std = torch.cat(pred_std, dim=0)
        # Move to cpu
        pred_mean = pred_mean.to("cpu")
        pred_std = pred_std.to("cpu")
        # Denormalize the prediction
        out_features_mean = self.statistics["out_features"]["mean"]
        out_features_std = self.statistics["out_features"]["std"]

        pred_mean = pred_mean.unsqueeze(-1) * out_features_std + out_features_mean
        pred_std = pred_std.unsqueeze(-1) * out_features_std

        # reshape
        pred_mean = pred_mean.reshape(LF_field.shape)
        pred_std = pred_std.reshape(LF_field.shape)

        # Compute the final prediction
        if self.floral:
            pred_residual = pred_mean
            pred_field = pred_residual + LF_field
        else:
            pred_field = pred_mean
            pred_residual = pred_mean - LF_field
        true_residual = HF_field - LF_field

        # Update the results to the dictionary
        field["LF_field"] = LF_field
        field["HF_field"] = HF_field
        field["Prediction"]["mean"] = pred_field
        field["Prediction"]["std"] = pred_std
        residual["True"] = true_residual
        residual["Prediction"]["mean"] = pred_residual
        residual["Prediction"]["std"] = pred_std

        # Save the results
        results = {
            "field": field,
            "residual": residual,
            "statistics": self.statistics,
            "domain": self.test_config["domain"],
            "job_name": self.job_name,
            "test_config": self.test_config,
        }

        # Save the results to a file
        path = self.job_name + "_GP_results"
        save_path = path + "_floral" if self.floral else path
        save_path += ".pt"
        printer("Saving results to {}".format(save_path))
        torch.save(
            results,
            save_path,
        )
