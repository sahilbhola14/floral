import math
import numpy as np
import lightning as L
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
from .utils_IO import (
    check_path,
    printer,
    check_tensor_blowup,
    n2t,
    check_keys,
    deep_get,
)


def build_data_module(
    config: dict,
    hp_config: wandb.sdk.wandb_config.Config | dict = None,
    verbose: bool = False,
):
    """Get the data module for the oneDCorr problem.
    Args:
        config (dict): Configuration dictionary containing data parameters.
        hp_config (dict): Hyperparameter configuration dictionary.
    """
    # create the data module
    data_module = OpDataModule(config=config, hp_config=hp_config, verbose=verbose)
    # Setup the data module
    data_module.setup()

    return data_module


class OpDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: dict,
        hp_config: wandb.sdk.wandb_config.Config | dict,
        verbose: bool = False,
    ):
        super(OpDataModule, self).__init__()
        self.config = config
        self.hp_config = hp_config

        # extract config
        self.floral = self.config.get("floral", False)
        self.LF_path = self.config.data.LF.path
        self.HF_path = self.config.data.HF.path
        self.n_samples = self.config.data.get("n_samples", 10)
        self.n_val = self.config.data.get("n_val", 10)
        self.n_train_all = self.n_samples - self.n_val
        assert (
            self.n_train_all > 0
        ), f"number of validation samples cannot exceed : {self.n_samples}"
        self.n_train = self.config.data.get("n_train", 10)
        assert (
            self.n_train <= self.n_train_all
        ), f"number of train samples must be less than (or equal) to {self.n_train_all}"
        self.dataloader_config = self.config.dataloader
        self.reload = self.dataloader_config.reload
        self.num_workers = self.dataloader_config.num_workers
        # extract hp_config
        self.batch_size = self.hp_config.get("batch_size", 64)

        # load the data
        printer(f"Loading low-fidelity data from {self.LF_path}")
        self.LF_data = self._load_data(path=self.LF_path)
        printer(f"Loading high-fidelity data from {self.HF_path}")
        self.HF_data = self._load_data(path=self.HF_path)
        printer("Done loading data")

        # file paths
        self.file_paths = self._get_file_paths()

        # verbose
        if verbose:
            self._print_header()

        assert (
            self.n_train + self.n_val <= self.n_samples
        ), "number of (train + val) samples cannot exceed available samples"

    def _load_data(self, path):
        """Load data from the specified path"""
        # initial check
        check_path(path)
        # load
        data = dict(np.load(path, allow_pickle=True))
        # load  check
        required_keys = [
            "field",
            "condition",
            "field_domain",
            "condition_domain",
        ]
        check_keys(data, required_keys)
        return data

    def _get_file_paths(self):
        """Get the file paths for saving/loading datasets and statistics"""
        # file paths
        file_paths = {
            "datasets": {
                "train": "trainset_floral.pt" if self.floral else "trainset_flora.pt",
                "val": "valset_floral.pt" if self.floral else "valset_flora.pt",
            },
            "statistics": (
                "statistics_floral.pt" if self.floral else "statistics_flora.pt"
            ),
        }
        return file_paths

    def _print_header(self):
        """print the header for the dataloader"""
        printer("==" * 50)
        printer("**" * 10 + "Dataloader config" + "**" * 10)
        printer(f"Reload datasets: {self.reload}")
        printer(f"Train samples: {self.n_train}")
        printer(f"Validation samples: {self.n_val}")
        printer(f"Train/Val ratio: {self.n_train/self.n_val}")
        printer(f"Batch size: {self.batch_size}")
        printer(f"Num workers: {self.num_workers}")
        printer(
            f"Normalize field: {self.dataloader_config.normalize.target_field.enabled}"
            f" with Auto: {self.dataloader_config.normalize.target_field.auto}"
        )
        printer(
            f"Normalize condition: {self.dataloader_config.normalize.condition.enabled}"
            f" with Auto: {self.dataloader_config.normalize.condition.auto}"
        )
        printer("==" * 50)

    def _extract_keys(self, data_dict):
        """extract the keys: field, condition, domain from the data_dict"""
        # extract dict
        field = data_dict.get("field")
        condition = data_dict.get("condition")
        field_domain = data_dict.get("field_domain")
        condition_domain = data_dict.get("condition_domain")

        # assert statements
        field_batch_size, field_channels, *field_dims = field.shape
        condition_batch_size, condition_channels, *condition_dims = condition.shape
        assert isinstance(
            field, np.ndarray
        ), f"Expected numpy array, got {type(field).__name__}"
        assert isinstance(
            condition, np.ndarray
        ), f"Expected numpy array, got {type(condition).__name__}"
        assert isinstance(
            field_domain, np.ndarray
        ), f"Expected numpy array, got {type(field_domain).__name__}"
        assert isinstance(
            condition_domain, np.ndarray
        ), f"Expected numpy array, got {type(condition_domain).__name__}"
        assert (
            field_batch_size == condition_batch_size
        ), "incorrect number of field and condition samples"
        assert all(
            [
                field_domain.ndim == 2,
                field_domain.shape[1] == len(field_dims),
            ]
        ), "field domain dims inconsistent with the field"
        assert all(
            [
                condition_domain.ndim == 2,
                condition_domain.shape[1] == len(condition_dims),
            ]
        ), "condition domain dims inconsistent with the condition"

        assert len(field_domain) == math.prod(
            field_dims
        ), "inconsistent field domain points"
        assert len(condition_domain) == math.prod(
            condition_dims
        ), "inconsistent condition domain points"

        # convert to float tensor
        field_tensor = n2t(field)
        condition_tensor = n2t(condition)
        field_domain_tensor = n2t(field_domain)
        condition_domain_tensor = n2t(condition_domain)

        # convert domain(s) shape to match field
        field_domain_tensor = field_domain_tensor.T.view(-1, *field_dims).unsqueeze(0)
        condition_domain_tensor = condition_domain_tensor.T.view(
            -1, *condition_dims
        ).unsqueeze(0)
        return (
            field_tensor,
            condition_tensor,
            field_domain_tensor,
            condition_domain_tensor,
        )

    def _get_operator_data_dict(self):
        """get the operator dictionary"""
        # extract LF data tensors
        (
            LF_field,
            LF_condition,
            LF_field_domain,
            LF_condition_domain,
        ) = self._extract_keys(data_dict=self.LF_data)
        # extract HF data tensors
        (
            HF_field,
            HF_condition,
            HF_field_domain,
            HF_condition_domain,
        ) = self._extract_keys(data_dict=self.HF_data)
        # assert statements
        assert (
            LF_field.shape == HF_field.shape
        ), "Low-fidelity and High-fidelity must be defined on the same domain"
        assert (
            LF_condition.shape == HF_condition.shape
        ), "Low-fidelity and High-fidelity must be defined on the same domain"
        assert (
            LF_field_domain.shape == HF_field_domain.shape
        ), "Low-fidelity and High-fidelity must be defined on the same field domain"
        assert (
            LF_condition_domain.shape == HF_condition_domain.shape
        ), "Low-fidelity and High-fidelity must be defined on the same condition domain"
        # create target field
        if self.floral:
            target_field = HF_field - LF_field
        else:
            target_field = HF_field
        # check availabe samples
        assert self.n_samples <= len(target_field), (
            f"Requested samples: {self.n_samples} > "
            f"available samples: {len(target_field)}"
        )
        # create operator data dict
        op_data_dict = {}
        op_data_dict["target_field"] = target_field[: self.n_samples]
        op_data_dict["condition"] = HF_condition[: self.n_samples]
        op_data_dict["LF_field"] = LF_field[: self.n_samples]
        # create data shape dict
        shape_dict = {}
        shape_dict["field"] = {
            "channels": target_field.shape[1],
            "dims": list(target_field.shape[2:]),
            "ndim": len(target_field.shape[2:]),
        }
        shape_dict["condition"] = {
            "channels": HF_condition.shape[1],
            "dims": list(HF_condition.shape[2:]),
            "ndim": len(HF_condition.shape[2:]),
        }
        shape_dict["field_domain"] = {
            "channels": HF_field_domain.shape[1],
            "dims": list(HF_field_domain.shape[2:]),
            "ndim": len(HF_field_domain.shape[2:]),
        }
        shape_dict["condition_domain"] = {
            "channels": HF_condition_domain.shape[1],
            "dims": list(HF_condition_domain.shape[2:]),
            "ndim": len(HF_condition_domain.shape[2:]),
        }
        # create domain dict
        domain_dict = {
            "field": HF_field_domain,
            "condition": HF_condition_domain,
        }
        return op_data_dict, shape_dict, domain_dict

    def _get_split_data_dict(self, op_data_dict):
        """split the operator data dict to train and validation"""
        train_data_dict = {}
        val_data_dict = {}
        for k in op_data_dict:
            # this splitting ensure that the same validation set (from end) is used
            train_data_dict[k] = op_data_dict[k][: self.n_train_all][: self.n_train]
            val_data_dict[k] = op_data_dict[k][-self.n_val :]
            assert len(train_data_dict[k]) == self.n_train
            assert len(val_data_dict[k]) == self.n_val
        return train_data_dict, val_data_dict

    def _get_statistics(self, data, normalize_config):
        """get the statistics"""
        # check keys
        required_keys = ["enabled", "auto", "mean", "std"]
        missing_keys = [k for k in required_keys if k not in normalize_config]
        assert len(missing_keys) == 0, f"Missing keys: {', '.join(missing_keys)}"
        # extract shape
        _, n_channels, *dims = data.shape
        n_dims = len(dims)

        # normalize
        if normalize_config.enabled:
            if normalize_config.auto:
                mean = data.mean(dim=(0, *range(2, data.ndim)), keepdim=True)
                std = data.std(dim=(0, *range(2, data.ndim)), keepdim=True)
            else:
                mean = normalize_config.mean
                std = normalize_config.std
                assert len(mean) == n_channels, "provide mean for each channel"
                assert len(std) == n_channels, "provide std for each channel"
                # broadcast
                mean = torch.tensor(mean).unsqueeze(0).view(1, -1, *([1] * n_dims))
                std = torch.tensor(std).unsqueeze(0).view(1, -1, *([1] * n_dims))
        else:
            # do not normlize
            mean = [0] * n_channels
            std = [1] * n_channels
            # broadcast
            mean = torch.tensor(mean).unsqueeze(0).view(1, -1, *([1] * n_dims))
            std = torch.tensor(std).unsqueeze(0).view(1, -1, *([1] * n_dims))
        # check shape
        assert mean.shape == (1, n_channels, *([1] * n_dims)), "incorrect mean shape"
        assert std.shape == (1, n_channels, *([1] * n_dims)), "incorrect std shape"
        return mean, std

    def _get_normalize_data_dict(self, train_data_dict, val_data_dict):
        """normalize the data using the training data statistics"""
        # initial checks
        normalize_keys = ["target_field", "condition"]
        check_keys(train_data_dict, normalize_keys)
        check_keys(val_data_dict, normalize_keys)

        statistics = {}
        train_norm_data_dict = {}
        val_norm_data_dict = {}
        for k in normalize_keys:
            assert k in train_data_dict, f"{k} missing in train data dict"
            assert k in val_data_dict, f"{k} missing in val data dict"
            # load the data
            train_data = train_data_dict[k]
            val_data = val_data_dict[k]
            # get the statistics
            normalize_config = self.dataloader_config.normalize[k]
            mean, std = self._get_statistics(
                train_data, normalize_config=normalize_config
            )
            # normalize training data
            train_norm_data_dict[k] = (train_data - mean) / std
            # normalize validation data
            val_norm_data_dict[k] = (val_data - mean) / std
            # check for nan and infs
            check_tensor_blowup(train_norm_data_dict[k], k + " (train)")
            check_tensor_blowup(val_norm_data_dict[k], k + " (val)")
            # save statistics
            statistics[k] = {}
            statistics[k]["mean"] = mean
            statistics[k]["std"] = std

        # add LF_field (without any normalization)
        train_norm_data_dict["LF_field"] = train_data_dict["LF_field"]
        val_norm_data_dict["LF_field"] = val_data_dict["LF_field"]

        return train_norm_data_dict, val_norm_data_dict, statistics

    def _set_attribute(self, name, val):
        """set the attribute"""
        setattr(self, name, val)

    def _get_attribute(self, name):
        """get the attribute"""
        return getattr(self, name)

    def denormalize_field(self, normal_field: torch.Tensor):
        """denormalize the field"""
        # extract field statistics
        field_mean = deep_get(self.statistics, ["target_field", "mean"])
        field_std = deep_get(self.statistics, ["target_field", "std"])
        # input check
        assert normal_field.ndim == field_mean.ndim, "invalid normal field"
        assert normal_field.ndim == field_std.ndim, "invalid normal field"
        # denormalize
        denormal_field = normal_field * field_std + field_mean

        return denormal_field

    def denormalize_condition(self, normal_condition: torch.Tensor):
        """denormalize the condition"""
        # extract field statistics
        condition_mean = deep_get(self.statistics, ["condition", "mean"])
        condition_std = deep_get(self.statistics, ["condition", "std"])
        # input check
        assert normal_condition.ndim == condition_mean.ndim, "invalid normal condition"
        assert normal_condition.ndim == condition_std.ndim, "invalid normal condition"
        # denormalize
        denormal_condition = normal_condition * condition_std + condition_mean

        return denormal_condition

    def setup(self, stage=None):
        """dataset setup"""
        # check reload
        if self.reload:
            raise NotImplementedError
        else:
            # prepare the operator fields
            op_data_dict, shape_dict, domain_dict = self._get_operator_data_dict()
            # set shape dict attribute
            self._set_attribute("shape_dict", shape_dict)
            # set domain dict attribute
            self._set_attribute("domain_dict", domain_dict)
            # split
            train_data_dict, val_data_dict = self._get_split_data_dict(op_data_dict)
            # normalize
            (
                train_norm_data_dict,
                val_norm_data_dict,
                statistics,
            ) = self._get_normalize_data_dict(
                train_data_dict=train_data_dict,
                val_data_dict=val_data_dict,
            )
            # set statistics attribute
            self._set_attribute("statistics", statistics)
            # create tensor dastasets for training and validation
            self.train_set = TensorDataset(
                train_norm_data_dict.get("target_field"),
                train_norm_data_dict.get("condition"),
                train_norm_data_dict.get("LF_field"),
            )
            self.val_set = TensorDataset(
                val_norm_data_dict.get("target_field"),
                val_norm_data_dict.get("condition"),
                val_norm_data_dict.get("LF_field"),
            )

    def train_dataloader(self):
        """Returns the training dataloader"""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Return the validation dataloader"""
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_config.num_workers,
            pin_memory=True,
        )
