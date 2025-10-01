import math
import numpy as np
import lightning as L
import wandb
import torch
from torch.utils.data import DataLoader, TensorDataset
from .utils_IO import check_path, printer, check_tensor_blowup, n2t


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
        self.dataloader_config = self.config.dataloader
        self.reload = self.dataloader_config.reload
        self.num_workers = self.dataloader_config.num_workers
        self.train_ratio = self.dataloader_config.get("train_ratio", 0.7)

        # extract hp_config
        self.batch_size = self.hp_config.get("batch_size", 64)

        # load the data
        self.LF_data = self._load_data(path=self.LF_path)
        self.HF_data = self._load_data(path=self.HF_path)

        # file paths
        self.file_paths = self._get_file_paths()

        # verbose
        if verbose:
            self._print_header()

    def _load_data(self, path):
        """Load data from the specified path"""
        check_path(path)
        data = np.load(path, allow_pickle=True)
        self._check_required_data_keys(data)
        return data

    def _check_required_data_keys(self, data):
        """Check if the data has the required keys"""
        required_keys = [
            "field",
            "condition",
            "domain",
        ]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Data must contain the key '{key}'")

    def _get_file_paths(self):
        """Get the file paths for saving/loading datasets and statistics"""
        # file paths
        file_paths = {
            "datasets": {
                "train": "trainset_floral.pt" if self.floral else "trainset.pt",
                "val": "valset_floral.pt" if self.floral else "valset.pt",
            },
            "statistics": "statistics_floral.pt" if self.floral else "statistics.pt",
        }
        return file_paths

    def _print_header(self):
        """print the header for the dataloader"""
        printer("==" * 50)
        printer("**" * 10 + "Dataloader config" + "**" * 10)
        printer(f"Reload datasets: {self.reload}")
        printer(f"Train/Val ratio: {self.train_ratio}")
        printer(f"Batch size: {self.batch_size}")
        printer(f"Num workers: {self.num_workers}")
        printer(
            f"Normalize field: {self.dataloader_config.normalize.field.enabled}"
            f"with Auto: {self.dataloader_config.normalize.field.auto}"
        )
        printer(
            f"Normalize condition: {self.dataloader_config.normalize.condition.enabled}"
            f"with Auto: {self.dataloader_config.normalize.condition.auto}"
        )
        printer("==" * 50)

    def _extract_keys(self, data_dict):
        """extract the keys: field, condition, domain from the data_dict"""
        field = data_dict.get("field")
        condition = data_dict.get("condition")
        domain = data_dict.get("domain")
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
            domain, np.ndarray
        ), f"Expected numpy array, got {type(domain).__name__}"
        assert (
            field_batch_size == condition_batch_size
        ), "incorrect number of field and condition samples"
        assert all(
            [
                domain.ndim == 2,
                domain.shape[1] == len(field_dims),
                domain.shape[1] == len(condition_dims),
            ]
        ), "domain dims inconsistent with the field and condition"
        assert len(domain) == math.prod(field_dims) and len(domain) == math.prod(
            condition_dims
        ), "inconsistent domain points"

        # convert to float tensor
        field_tensor = n2t(field)
        condition_tensor = n2t(condition)
        domain_tensor = n2t(domain)

        # convert domain shape to match field
        domain_tensor = domain_tensor.T.view(-1, *field_dims).unsqueeze(0)

        return field_tensor, condition_tensor, domain_tensor

    def _get_operator_data_dict(self):
        """get the operator dictionary"""
        # extract LF data tensors
        LF_field, LF_condition, LF_domain = self._extract_keys(data_dict=self.LF_data)
        # extract HF data tensors
        HF_field, HF_condition, HF_domain = self._extract_keys(data_dict=self.HF_data)
        # assert statements
        assert (
            LF_field.shape == HF_field.shape
        ), "Low-fidelity and High-fidelity must be defined on the same domain"
        assert (
            LF_condition.shape == HF_condition.shape
        ), "Low-fidelity and High-fidelity must be defined on the same domain"
        assert (
            LF_domain.shape == HF_domain.shape
        ), "Low-fidelity and High-fidelity must be defined on the same domain"
        # create target field
        if self.floral:
            target_field = HF_field - LF_field
        else:
            target_field = HF_field

        # check availabe samples
        assert self.n_samples <= len(
            target_field
        ), f"Requested samples: {self.n_samples} > "
        f"available samples: {len(target_field)}"
        # create operator data dict
        op_data_dict = {}
        op_data_dict["target_field"] = target_field[: self.n_samples]
        op_data_dict["condition"] = HF_condition[: self.n_samples]
        op_data_dict["LF_field"] = LF_field[: self.n_samples]
        # create data shape dict
        shape_dict = {}
        shape_dict["target_field"] = {
            "channels": target_field.shape[1],
            "dims": list(target_field.shape[2:]),
            "ndim": len(target_field.shape[2:]),
        }
        shape_dict["condition"] = {
            "channels": HF_condition.shape[1],
            "dims": list(HF_condition.shape[2:]),
            "ndim": len(HF_condition.shape[2:]),
        }
        shape_dict["domain"] = {
            "channels": HF_domain.shape[1],
            "dims": list(HF_domain.shape[2:]),
            "ndim": len(HF_domain.shape[2:]),
        }
        # create domain dict
        domain_dict = {
            "target_field": HF_domain,
            "condition": HF_domain,
        }
        return op_data_dict, shape_dict, domain_dict

    def _get_split_data_dict(self, op_data_dict):
        """split the operator data dict to train and validation"""
        train_data_dict = {}
        val_data_dict = {}
        n_train = int(self.dataloader_config.train_ratio * self.n_samples)
        for k in op_data_dict:
            train_data_dict[k] = op_data_dict[k][:n_train]
            val_data_dict[k] = op_data_dict[k][n_train:]
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
        normalize_keys = ["target_field", "condition"]
        statistics = {}
        train_norm_data_dict = {}
        val_norm_data_dict = {}
        for k in normalize_keys:
            assert (
                k in train_data_dict and k in val_data_dict
            ), f"{k} not found in train and val data dict"
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

        # add LF_field
        train_norm_data_dict["LF_field"] = train_data_dict["LF_field"]
        val_norm_data_dict["LF_field"] = val_data_dict["LF_field"]

        return train_norm_data_dict, val_norm_data_dict, statistics

    def _set_attribute(self, name, val):
        """set the attribute"""
        setattr(self, name, val)

    def _get_attribute(self, name):
        """get the attribute"""
        return getattr(self, name)

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
