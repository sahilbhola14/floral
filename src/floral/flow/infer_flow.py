# src/floral/flow/infer_flow.py
"""
Infer the flow
"""
import torch
from tqdm import tqdm
import lightning as L
from .flow import Flow
from contextlib import nullcontext
from torch.amp import autocast
from floral.utils import print_section, check_path, printer
from omegaconf import DictConfig


def perform_inference(
    best_model_path: str, data_module: L.LightningDataModule, config: DictConfig | dict
):
    """perform inference"""
    print_section("Inference")
    # check best model path
    check_path(best_model_path)
    # load from check point
    best_flow = Flow.load_from_checkpoint(
        best_model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    # set to eval mode and compile
    best_flow.eval()
    if hasattr(best_flow, "compile") and torch.cuda.is_available():
        # Use torch.compile for PyTorch 2.0+ (significant speedup)
        printer("Compiling the model...")
        best_flow = torch.compile(best_flow, mode="max-autotune")
    # create inference object
    infer = Inference(
        best_flow=best_flow,
        data_module=data_module,
        job_name=config.job_name,
        floral=config.floral,
        generate_config=config.generate,
    )

    infer()


class Inference:
    def __init__(
        self,
        best_flow: L.LightningModule,
        data_module: L.LightningDataModule,
        job_name: str,
        floral: bool,
        generate_config: dict,
    ):
        # set attributes
        self.best_flow = best_flow
        assert self.best_flow.training is False, "model should be in training mode"
        self.device = self.best_flow.device
        self.job_name = job_name
        self.floral = floral
        self.generate_config = generate_config
        self.data_module = data_module
        # enable inference optimizations
        torch.backends.cudnn.benchmark = True
        self.use_amp = False
        # extract generate config
        self.n_gen = self.generate_config.get("n_gen", 10)
        self.minibatch_size = self.generate_config.get("minibatch_size", 100)
        # prepare inference input dict
        self.inference_input_dict = self._get_inference_input_dict(
            val_set=data_module.val_set
        )
        # domain dict
        self.domain_dict = self.data_module.domain_dict

    def _get_inference_input_dict(self, val_set):
        """extract the inference input"""
        assert len(val_set.tensors) == 3, "expected (target_field, condition, LF_field)"
        # extract
        target_field, condition, LF_field_plot = val_set.tensors
        # compute HF_field
        target_field_denormalize = self.data_module.denormalize_field(
            normal_field=target_field
        )
        if self.floral:
            HF_field_plot = target_field_denormalize + LF_field_plot
        else:
            HF_field_plot = target_field_denormalize
        # compute condition_plot
        condition_plot = self.data_module.denormalize_condition(
            normal_condition=condition
        )
        # create dict
        inference_input_dict = {
            "condition": condition,
            "condition_plot": condition_plot,
            "HF_field_plot": HF_field_plot,
            "LF_field_plot": LF_field_plot,
            "n_batches": int(
                (len(val_set) + self.minibatch_size - 1) / self.minibatch_size
            ),
        }
        return inference_input_dict

    @torch.no_grad()
    def __call__(self):
        """perform inference"""
        # local attributes
        n_batches = self.inference_input_dict.get("n_batches")
        autocast_context = autocast("cuda") if self.use_amp else nullcontext()
        pbar = tqdm(
            range(n_batches),
            desc="Inference",
            ncols=150,
            leave=True,
        )
        all_LF_field_plot = self.inference_input_dict.get("LF_field_plot")
        all_condition = self.inference_input_dict.get("condition")
        # iterate
        all_prediction_plot = []
        for batch_idx in pbar:
            lower_idx = batch_idx * self.minibatch_size
            upper_idx = (batch_idx + 1) * self.minibatch_size
            with autocast_context:
                # get batches
                batch_LF_field_plot = all_LF_field_plot[lower_idx:upper_idx]
                batch_condition = all_condition[lower_idx:upper_idx]
                # get prediction
                batch_prediction_plot = self._get_prediction(condition=batch_condition)
                if self.floral:
                    batch_prediction_plot += batch_LF_field_plot.unsqueeze(1)
                all_prediction_plot.append(batch_prediction_plot)

        # ensure all async transfers are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # close the bar
        pbar.close()

        # stack
        all_prediction_plot = torch.vstack(all_prediction_plot)

        # create result dict
        result_dict = {
            "HF_field_plot": self.inference_input_dict.get("HF_field_plot"),
            "LF_field_plot": self.inference_input_dict.get("LF_field_plot"),
            "condition_plot": self.inference_input_dict.get("condition_plot"),
            "HF_field_prediction_plot": all_prediction_plot,
            "domain_dict": self.data_module.domain_dict,
        }
        # save the results to a file
        save_path = f"{self.job_name}_results{'_floral' if self.floral else ''}.pt"
        printer(f"Saving results to {save_path}")
        torch.save(result_dict, save_path)

    def _get_prediction(self, condition: torch.Tensor):
        """integrate the ODE"""
        # normalized prediciton
        prediction = self.best_flow.integrate_flow(
            condition=condition,
            **self.generate_config,
        ).cpu()
        # denormalize
        batch_size, n_gen, *dims = prediction.shape
        prediction_plot = self.data_module.denormalize_field(
            normal_field=prediction.view(batch_size * n_gen, *dims)
        ).view(batch_size, n_gen, *dims)
        return prediction_plot
