# src/floral/flow/infer_flow.py
"""
Inference module
Perform the inference on the testset
"""
import torch
from tqdm import tqdm
import lightning as L
from .flow import Flow
from contextlib import nullcontext
from torch.amp import autocast
from floral.utils import print_section, check_path, printer
from omegaconf import DictConfig, OmegaConf
from collections.abc import Mapping
from dataclasses import dataclass, field


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
    printer(f"Loaded from checkpoint: {best_model_path}")
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
        train_res=config.train.train_res,
        operator_method=config.flow.operator.method,
    )
    # perform inference
    infer()


# inference class config
@dataclass
class InferenceConfig:
    """Default config for Inference with override support."""

    generate: dict = field(
        default_factory=lambda: {
            "n_gen": 10,
            "n_steps": 2,
            "method": "dopri5",
            "rtol": 1e-5,
            "atol": 1e-5,
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

    def resolve(self, generate_config=None):
        merged = OmegaConf.merge(
            OmegaConf.create(self.generate),
            self._to_dict_config(generate_config),
        )
        return merged


class Inference:
    def __init__(
        self,
        best_flow: L.LightningModule,
        data_module: L.LightningDataModule,
        job_name: str,
        floral: bool,
        generate_config: dict,
        train_res: str | int,
        operator_method: str,
    ):
        # resolve generate config against defaults
        infer_config_defaults = InferenceConfig()
        self.generate_config = infer_config_defaults.resolve(generate_config)

        # set attributes
        self.best_flow = best_flow
        assert self.best_flow.training is False, "model should be in training mode"
        self.device = self.best_flow.device
        self.job_name = job_name.lower().strip()
        self.floral = floral
        self.operator_method = operator_method

        # enable inference optimizations
        torch.backends.cudnn.benchmark = True
        self.use_amp = False

        self.data_module = data_module
        self.train_res = (
            (train_res).strip().lower() if isinstance(train_res, str) else train_res
        )

        # extract test set
        self.data_module = data_module
        self.test_loader = data_module.test_dataloader()

    @torch.no_grad()
    def __call__(self):
        """perform inference"""
        # local attributes
        autocast_context = autocast("cuda") if self.use_amp else nullcontext()

        # iterate
        all_HF_field = []
        all_LF_field = []
        all_condition = []
        all_prediction = []

        for batch in tqdm(self.test_loader, desc="Inference", ncols=150, leave=True):
            with autocast_context:
                # extract
                target_field = batch.get("target_field")
                condition = batch.get("condition")
                LF_field = batch.get("LF_field")
                # denormalize inputs
                target_field = self.data_module.denormalize_field(target_field).cpu()
                condition = self.data_module.denormalize_condition(condition).cpu()
                LF_field = self.data_module.denormalize_LF_field(LF_field).cpu()

                # get denormalized prediction (pass original normalized batch tensors)
                prediction = self._get_prediction(
                    condition=batch.get("condition"), LF_field=batch.get("LF_field")
                )
                if self.floral:
                    prediction = prediction + LF_field.unsqueeze(1)

                all_HF_field.append(target_field)
                all_LF_field.append(LF_field)
                all_condition.append(condition)
                all_prediction.append(prediction)

        # ensure all async transfers are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # stack across batches
        all_HF_field = torch.cat(all_HF_field, dim=0)
        all_LF_field = torch.cat(all_LF_field, dim=0)
        all_condition = torch.cat(all_condition, dim=0)
        all_prediction = torch.cat(all_prediction, dim=0)

        result_dict = {
            "HF_field": all_HF_field,
            "LF_field": all_LF_field,
            "condition": all_condition,
            "prediction": all_prediction,
        }

        # save the results to a file
        save_path_identifier = "floral" if self.floral else "flora"
        job_identifier = (
            self.job_name
            + f"_operator_method_{self.operator_method.strip().lower()}"
            + f"_n_train_{self.data_module.n_train}"
            + f"_n_val_{self.data_module.n_val}"
            + f"_n_test_{self.data_module.n_test}"
            + f"_train_res_{self.train_res}"
        )
        save_path = f"{job_identifier}_results_{save_path_identifier}.pt"
        printer(f"Saving results to {save_path}")
        torch.save(result_dict, save_path)

    def _get_prediction(self, condition: torch.Tensor, LF_field: torch.Tensor):
        """integrate the ODE and return the de-normalized prediciton"""
        # integrate the ode (normalized)
        prediction = self.best_flow.integrate_flow(
            condition=condition,
            LF_field=LF_field,
            generate_config=self.generate_config,
        ).cpu()

        # denormalize
        batch_size, n_gen, *dims = prediction.shape

        prediction = self.data_module.denormalize_field(
            normal_field=prediction.view(batch_size * n_gen, *dims)
        ).view(batch_size, n_gen, *dims)

        return prediction
