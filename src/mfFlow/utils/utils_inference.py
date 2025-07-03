import torch
import pytorch_lightning as L
from mfFlow.utils import printer


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

        for ii in range(n_samples):
            printer("Pecentage: {:.2f}%".format(ii / n_samples * 100))
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
        torch.save(
            results,
            self.job_name + "_results" + "_mfFlow.pt" if self.mfFlow else "_results.pt",
        )

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
