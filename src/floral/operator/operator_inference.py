# src/floral/flow/operator_inference.py
"""
Inference module
Perform the inference on the testset
"""
import time
import torch
from tqdm import tqdm
import lightning as L
from .operator import Operator
from contextlib import nullcontext
from torch.amp import autocast
from floral.utils import print_section, check_path, printer
from omegaconf import DictConfig


def operator_inference(
    best_model_path: str, data_module: L.LightningDataModule, config: DictConfig | dict
):
    """perform inference"""
    print_section("Inference")
    # check best model path
    check_path(best_model_path)
    # load from check point
    best_operator = Operator.load_from_checkpoint(
        best_model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    printer(f"Loaded from checkpoint: {best_model_path}")
    # set to eval mode and compile
    best_operator.eval()
    if hasattr(best_operator, "compile") and torch.cuda.is_available():
        # Use torch.compile for PyTorch 2.0+ (significant speedup)
        printer("Compiling the model...")
        best_operator = torch.compile(best_operator, mode="max-autotune")

    # create inference object
    infer = Inference(
        best_operator=best_operator,
        data_module=data_module,
        job_name=config.job_name,
        floral=config.floral,
        train_res=config.train.train_res,
        operator_method=config.flow.operator.method,
    )
    # perform inference
    infer()


class Inference:
    def __init__(
        self,
        best_operator: L.LightningModule,
        data_module: L.LightningDataModule,
        job_name: str,
        floral: bool,
        train_res: str | int,
        operator_method: str,
    ):
        # set attributes
        self.best_operator = best_operator
        assert self.best_operator.training is False, "model should be in training mode"
        self.device = self.best_operator.device
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
        n_fields = self.data_module.n_fields_per_sample
        n_unique_test = self.data_module.n_unique_test

        # iterate
        all_HF_field = []
        all_LF_field = []
        all_condition = []
        all_prediction = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_infer_start = time.perf_counter()

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
                    # prediction field = residual_prediction + LF_field
                    prediction = prediction + LF_field.unsqueeze(1)
                    # target field = residual + LF_field
                    HF_field = target_field + LF_field
                else:
                    # target field = HF_field
                    HF_field = target_field

                all_HF_field.append(HF_field)
                all_LF_field.append(LF_field)
                all_condition.append(condition)
                all_prediction.append(prediction)

        # ensure all async transfers are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_time_s = time.perf_counter() - t_infer_start
        time_per_condition_s = inference_time_s / n_unique_test
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else None
        )
        printer(
            f"Inference time: {inference_time_s:.2f} s | "
            f"Time per condition: {time_per_condition_s:.4f} s"
            + (
                f" | Peak GPU memory: {peak_gpu_memory_mb:.1f} MB"
                if peak_gpu_memory_mb
                else ""
            )
        )

        # stack across batches — flat shapes:
        #   HF/LF:      (n_test, C, *dims)   where n_test = n_unique_test * n_fields
        #   prediction: (n_test, 1, C, *dims)
        all_HF_field = torch.cat(all_HF_field, dim=0)
        all_LF_field = torch.cat(all_LF_field, dim=0)
        all_condition = torch.cat(all_condition, dim=0)
        all_prediction = torch.cat(all_prediction, dim=0)

        # Tiling layout from _condition_rows (sorted):
        #   [c0_f0, c1_f0, ..., cN_f0, c0_f1, ..., cN_fK]
        # view(n_fields, n_unique_test, ...)
        # then permute → (n_unique_test, n_fields, C, *dims)
        field_shape = all_HF_field.shape[1:]  # (C, *dims)
        ndim = len(field_shape)
        perm = (1, 0, *range(2, 2 + ndim))

        all_HF_field = (
            all_HF_field.view(n_fields, n_unique_test, *field_shape)
            .permute(*perm)
            .contiguous()
        )  # (n_unique_test, n_fields, C, *dims)
        all_LF_field = (
            all_LF_field.view(n_fields, n_unique_test, *field_shape)
            .permute(*perm)
            .contiguous()
        )  # (n_unique_test, n_fields, C, *dims)

        # unique conditions — all realizations share the same condition, take block 0
        all_condition = all_condition[
            :n_unique_test
        ]  # (n_unique_test, C_cond, *cond_dims)

        # reshape prediction: (n_test, 1, C, *dims)
        #   → (n_fields, n_unique_test, 1, C, *dims)  [block layout]
        #   → (n_unique_test, n_fields, 1, C, *dims)  [permute]
        #   → (n_unique_test, n_fields, C, *dims)     [flatten realization × gen dims]
        n_gen = all_prediction.shape[1]
        pred_shape = all_prediction.shape[2:]  # (C, *dims)
        pred_ndim = len(pred_shape)
        pred_perm = (1, 0, 2, *range(3, 3 + pred_ndim))
        all_prediction = (
            all_prediction.view(n_fields, n_unique_test, n_gen, *pred_shape)
            .permute(*pred_perm)
            .contiguous()
            .view(n_unique_test, n_fields * n_gen, *pred_shape)
        )  # (n_unique_test, n_fields * n_gen, C, *dims)

        # repeat HF/LF along dim=1 to match prediction: n_fields → n_fields * n_gen
        all_HF_field = all_HF_field.repeat_interleave(n_gen, dim=1)
        all_LF_field = all_LF_field.repeat_interleave(n_gen, dim=1)
        # both now: (n_unique_test, n_fields * n_gen, C, *dims)

        result_dict = {
            "HF_field": all_HF_field,
            "LF_field": all_LF_field,
            "condition": all_condition,
            "prediction": all_prediction,
            "domain_dict": self.data_module.domain_dict,
            "n_unique_train": self.data_module.n_unique_train,
            "n_unique_val": self.data_module.n_unique_val,
            "n_unique_test": self.data_module.n_unique_test,
            "n_unique_conditions": self.data_module.n_unique_conditions,
            "n_fields_per_sample": self.data_module.n_fields_per_sample,
            "n_train": self.data_module.n_train,
            "n_val": self.data_module.n_val,
            "n_test": self.data_module.n_test,
            # computational cost (batch-size independent)
            "inference_time_s": inference_time_s,
            "time_per_condition_s": time_per_condition_s,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        }

        # save the results to a file
        save_path_identifier = "floral" if self.floral else "flora"
        job_identifier = (
            self.job_name
            + f"_fno_operator_method_{self.operator_method.strip().lower()}"
            + f"_n_train_{self.data_module.n_train}"
            + f"_n_val_{self.data_module.n_val}"
            + f"_n_test_{self.data_module.n_test}"
            + f"_train_res_{self.train_res}"
        )
        save_path = f"{job_identifier}_results_{save_path_identifier}.pt"
        printer(f"Saving results to {save_path}")
        torch.save(result_dict, save_path)

    def _get_prediction(self, condition: torch.Tensor, LF_field: torch.Tensor):
        """forward pass through the model"""
        # prediction (normalized)
        prediction = self.best_operator._forward_operator(
            condition=condition.to(self.device),
            LF_field=LF_field.to(self.device),
        ).cpu()

        # denormalize
        batch_size, *dims = prediction.shape
        prediction = self.data_module.denormalize_field(normal_field=prediction)

        return prediction.unsqueeze(1)
