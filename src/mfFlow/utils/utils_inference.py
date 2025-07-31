import torch
import pytorch_lightning as L
import gpytorch
import warnings
from mfFlow.utils import printer
from tqdm import tqdm
from gpytorch.models import ExactGP
from contextlib import nullcontext
from torch.amp import autocast


class OptimizedInference:
    """Inference class for handling inference operations."""

    def __init__(
        self,
        model: L.LightningModule,
        test_config: dict,
        statistics: dict,
        job_name: str,
        mfFlow: bool,
        generate_config: dict,
    ):
        # Initialize the model with the best_model_path
        self.model = model
        # Set the model to evaluation mode
        self.model.eval()
        # Enable inference optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes

        # Store the test configuration, statistics, job name, and number of samples
        self.test_config = test_config  # Configuration for the inference task
        self.statistics = statistics  # Statistics of the data
        self.job_name = job_name  # Name of the inference job
        self.mfFlow = mfFlow  # Flag for multi-flow processing
        self.generate_config = generate_config  # Configuration for generation settings

        # Extract generate config
        self.minibatch_size = generate_config.get("minibatch_size", None)
        self.nT = generate_config.get("nT", None)  # Number of time steps
        self.n_gen = generate_config.get(
            "n_gen", None
        )  # Number of generations (for UQ)

        # Pre-comptue normalization facotrs
        self.field_mean = self.statistics["field"]["mean"].item()
        self.field_std = self.statistics["field"]["std"].item()

        # Pre-move the domain to device
        self.d_eval = self.test_config["domain"].to(self.model.device)

    @torch.no_grad()
    def __call__(self):
        """Perform the inference task"""
        n_samples = self.test_config.get("n_samples")

        # Pre-allocate tensors
        device = self.model.device
        field = {
            "LF_field": torch.empty(
                n_samples, *self.test_config["LF_field"][0].shape, device="cpu"
            ),
            "HF_field": torch.empty(
                n_samples, *self.test_config["HF_field"][0].shape, device="cpu"
            ),
            "Prediction": torch.empty(
                n_samples, self.n_gen, len(self.d_eval), device="cpu"
            ),
        }
        residual = {
            "True": torch.empty(
                n_samples, *self.test_config["LF_field"][0].shape, device="cpu"
            ),
            "Prediction": torch.empty(
                n_samples, self.n_gen, len(self.d_eval), device="cpu"
            ),
        }

        # use mixed precision if available for faster inference
        use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, "autocast")
        printer(f"Using automatic mixed-precision: {use_amp}")
        autocast_context = autocast("cuda") if use_amp else nullcontext()

        pbar = tqdm(
            range(n_samples),
            desc="Inference",
            ncols=150,
            leave=True,
        )

        for ii in pbar:
            with autocast_context:
                # get the condition
                c_eval = (
                    self.test_config["condition"][ii]
                    .unsqueeze(0)
                    .to(device, non_blocking=True)
                )

                # true field
                LF_field = self.test_config["LF_field"][ii]
                HF_field = self.test_config["HF_field"][ii]
                true_residual = HF_field - LF_field

                # perform the prediction
                if self.mfFlow:
                    # pred_residual = self._get_prediction(c_eval)
                    pred_residual = self._get_prediction(c_eval)
                    # Denormalize in-place
                    pred_residual = pred_residual * self.field_std + self.field_mean
                    # Get the high fidelity field prediction
                    pred_field = pred_residual + LF_field.unsqueeze(0).to(device)
                    pred_field = pred_field.to(
                        "cpu", non_blocking=True
                    )  # Move back to CPU
                    pred_residual = pred_residual.to("cpu", non_blocking=True)
                else:
                    pred_field = self._get_prediction(c_eval)
                    # Denormalize in-place
                    pred_field = pred_field * self.field_std + self.field_mean
                    pred_field_cpu = pred_field.to("cpu", non_blocking=True)
                    # Compute the residual
                    pred_residual = (pred_field - LF_field.unsqueeze(0).to(device)).to(
                        "cpu", non_blocking=True
                    )
                    pred_field = pred_field_cpu

            # Store results
            field["LF_field"][ii] = LF_field
            field["HF_field"][ii] = HF_field
            field["Prediction"][ii] = (
                pred_field.squeeze(0) if pred_field.dim() > 2 else pred_field
            )
            residual["True"][ii] = true_residual
            residual["Prediction"][ii] = (
                pred_residual.squeeze(0) if pred_residual.dim() > 2 else pred_residual
            )

        # Ensure all async transfers are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # check for nans
        if torch.isnan(field["Prediction"]).any():
            warnings.warn("Nans found in prediction.", RuntimeWarning)

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
        save_path = f"{self.job_name}_results{'_mfFlow' if self.mfFlow else ''}.pt"
        printer(f"Saving results to {save_path}")
        torch.save(results, save_path)
        pbar.close()

    def _get_prediction(self, c_eval: torch.Tensor):
        """Get the model prediction for a given conditon over the entire domain"""
        total_size = len(self.d_eval)
        pred = torch.empty(
            self.n_gen, total_size, device=c_eval.device, dtype=torch.float32
        )  # adjust dtype as needed

        start_idx = 0

        for ii in range(0, total_size, self.minibatch_size):
            d_batch = self.d_eval[ii : ii + self.minibatch_size]

            batch_pred = (
                self.model.interpolate(
                    c_eval=c_eval,
                    d_eval=d_batch,
                    n_gen=self.n_gen,
                    nT=self.nT,
                    method="dopri5",
                    atol=1e-4,
                    rtol=1e-4,
                    max_n_gen_per_batch=20,
                )
                .squeeze(-1)
                .T.detach()
            )  # shape: (n_gen, batch_size)

            batch_size = batch_pred.shape[1]
            pred[
                :, start_idx : start_idx + batch_size
            ] = batch_pred  # copy batch into pre-allocated tensor
            start_idx += batch_size

        assert pred.shape == (self.n_gen, total_size)

        return pred


class Inference:
    """Inference class for handling inference operations."""

    def __init__(
        self,
        model: L.LightningModule,
        test_config: dict,
        statistics: dict,
        job_name: str,
        mfFlow: bool,
        generate_config: dict,
    ):
        # Initialize the model with the best_model_path
        self.model = model
        # Set the model to evaluation mode
        self.model.eval()
        # Store the test configuration, statistics, job name, and number of samples
        self.test_config = test_config  # Configuration for the inference task
        self.statistics = statistics  # Statistics of the data
        self.job_name = job_name  # Name of the inference job
        self.mfFlow = mfFlow  # Flag for multi-flow processing
        self.generate_config = generate_config  # Configuration for generation settings
        # Extract generate config
        self.minibatch_size = generate_config.get("minibatch_size", None)
        self.nT = generate_config.get("nT", None)  # Number of time steps
        self.n_gen = generate_config.get(
            "n_gen", None
        )  # Number of generations (for UQ)

    def __call__(self):
        """Perform the inference task"""
        n_samples = self.test_config.get("n_samples")
        field = {"LF_field": [], "HF_field": [], "Prediction": []}
        residual = {"True": [], "Prediction": []}
        pbar = tqdm(
            range(n_samples),
            desc="Inference",
            ncols=150,
            leave=True,
        )
        for ii in pbar:
            # Get the condition
            c_eval = (
                self.test_config["condition"][ii].unsqueeze(0).to(self.model.device)
            )
            # Get the domain
            d_eval = self.test_config["domain"].to(self.model.device)
            # True fields
            LF_field = self.test_config["LF_field"][ii]
            HF_field = self.test_config["HF_field"][ii]
            true_residual = (HF_field - LF_field).unsqueeze(0)
            # Perform the prediction
            if self.mfFlow:
                # compute the prediction for multi-flow
                pred_residual = self._get_prediction(c_eval, d_eval)
                # denormalize the prediction
                pred_residual = (
                    pred_residual * self.statistics["field"]["std"].item()
                    + self.statistics["field"]["mean"].item()
                )
                # get the high fidelity field prediction
                pred_field = pred_residual + LF_field.unsqueeze(0)
            else:
                # compute the prediction for single-flow
                pred_field = self._get_prediction(c_eval, d_eval)
                # denormalize the prediction
                pred_field = (
                    pred_field * self.statistics["field"]["std"].item()
                    + self.statistics["field"]["mean"].item()
                )
                # compute the residual
                pred_residual = pred_field - LF_field.unsqueeze(0)

            # updated the results to the dictionary
            field["LF_field"].append(LF_field.unsqueeze(0))
            field["HF_field"].append(HF_field.unsqueeze(0))
            field["Prediction"].append(pred_field)
            residual["True"].append(true_residual)
            residual["Prediction"].append(pred_residual)

        # Convert lists to tensors
        field["LF_field"] = torch.vstack(field["LF_field"])
        field["HF_field"] = torch.vstack(field["HF_field"])
        field["Prediction"] = torch.stack(field["Prediction"])
        residual["True"] = torch.vstack(residual["True"])
        residual["Prediction"] = torch.stack(residual["Prediction"])

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
        path = self.job_name + "_results"
        save_path = path + "_mfFlow" if self.mfFlow else path
        save_path += ".pt"
        printer("Saving results to {}".format(save_path))
        torch.save(
            results,
            save_path,
        )

        pbar.close()

    def _get_prediction(self, c_eval: torch.Tensor, d_eval: torch.Tensor):
        """Get the model prediction"""
        pred = []
        for ii in range(0, len(d_eval), self.minibatch_size):
            d_batch = d_eval[ii : ii + self.minibatch_size]
            batch_pred = (
                self.model.interpolate(
                    c_eval=c_eval,
                    d_eval=d_batch,
                    n_gen=self.n_gen,
                    nT=self.nT,
                )
                .squeeze(-1)
                .T.detach()
                .to("cpu")
            )
            pred.append(batch_pred)

        pred = torch.hstack(pred)

        assert pred.shape == (self.n_gen, len(d_eval))

        return pred


class InferenceGP:
    """Inference clas for Gaussian Process"""

    def __init__(
        self,
        model: ExactGP,
        test_config: dict,
        statistics: dict,
        job_name: str,
        mfFlow: bool,
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
        self.mfFlow = mfFlow  # Flag for multi-flow processing
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
        if self.mfFlow:
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
        save_path = path + "_mfFlow" if self.mfFlow else path
        save_path += ".pt"
        printer("Saving results to {}".format(save_path))
        torch.save(
            results,
            save_path,
        )
