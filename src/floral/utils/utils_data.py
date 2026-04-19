import math
import lightning as L
import wandb
import torch
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from collections.abc import Mapping
from omegaconf import DictConfig, OmegaConf
from .utils_IO import (
    check_path,
    printer,
    print_section,
    check_tensor_blowup,
    check_keys,
    deep_get,
    type_check,
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


class MultiFidelityDataLoader(Dataset):
    """Custom Dataset"""

    def __init__(self, data_dict):
        """
        Args:
            data_dict: dict like
            {
                "target_field":  torch.tensor,
                "condition":  torch.tensor,
                "LF_field":  torch.tensor,
            }
        """
        self.data_dict = data_dict

        # sanity check
        self.length = next(iter(data_dict.values())).shape[0]
        self._check_consistency()

    def _check_consistency(self):
        """check the length size"""
        for key in self.data_dict:
            assert self.data_dict[key].shape[0] == self.length, f"Mismatch in {key}"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """sample: dict like
        {
            "target_field":  tensor
            "condition": tensor
            "LF_Field": tensor,
        }
        """
        sample = {k: self.data_dict[k][idx] for k in self.data_dict}

        return sample


@dataclass
class OpDataModuleConfig:
    """Default config for DataModule with override support."""

    # config default
    config: dict = field(
        default_factory=lambda: {
            "floral": False,
            "data": {
                "LF": {"path": "low_fidelity.pt"},
                "HF": {"path": "high_fidelity.pt"},
            },
            "dataloader": {
                "reload": False,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "n_train_samples": None,  # if set, overrides train_ratio for train size
                "num_workers": 1,
                "normalize": {
                    "target_field": {
                        "enabled": True,
                        "auto": True,
                        "mean": [0.0],
                        "std": [1.0],
                    },
                    "condition": {
                        "enabled": True,
                        "auto": True,
                        "mean": [0.0],
                        "std": [1.0],
                    },
                    "LF_field": {
                        "enabled": False,
                        "auto": True,
                        "mean": [0.1],
                        "std": [0.5],
                    },
                },
            },
        }
    )

    # hp config default
    hp_config: dict = field(default_factory=lambda: {"batch_size": 32})

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

    def resolve(self, config=None, hp_config=None):
        merged_config = OmegaConf.merge(
            OmegaConf.create(self.config),
            self._to_dict_config(config),
        )
        merged_hp_config = OmegaConf.merge(
            OmegaConf.create(self.hp_config),
            self._to_dict_config(hp_config),
        )
        return merged_config, merged_hp_config


class OpDataModule(L.LightningDataModule):
    def __init__(
        self,
        config: dict,
        hp_config: wandb.sdk.wandb_config.Config | dict,
        verbose: bool = False,
    ):
        super(OpDataModule, self).__init__()
        # merge defaults with user-provided runtime config/hp_config
        config_defaults = OpDataModuleConfig()
        self.config, self.hp_config = config_defaults.resolve(
            config=config,
            hp_config=hp_config,
        )

        # extract config
        self.floral = self.config.get("floral")
        self.LF_path = deep_get(self.config, ["data", "LF", "path"])
        self.HF_path = deep_get(self.config, ["data", "HF", "path"])

        self.dataloader_config = self.config.dataloader
        self.reload = self.dataloader_config.get("reload")
        self.num_workers = self.dataloader_config.get("num_workers")
        self.train_ratio = self.dataloader_config.get("train_ratio")
        self.val_ratio = self.dataloader_config.get("val_ratio")
        self.test_ratio = self.dataloader_config.get("test_ratio")
        # optional override: fix train size independently of train_ratio
        self.n_train_samples = self.dataloader_config.get("n_train_samples")

        # extract hp_config
        self.batch_size = self.hp_config.get("batch_size", 64)

        # load the data
        self.LF_data = self._load_data(path=self.LF_path, verbose=verbose)
        self.HF_data = self._load_data(path=self.HF_path, verbose=verbose)

        # unique-condition metadata
        # (tile layout: condition i at rows i, i+n_unique, ...)
        self.n_unique_conditions = int(self.HF_data["n_unique_conditions"])
        self.n_fields_per_sample = int(self.HF_data["n_fields_per_sample"])

        # file paths
        self.file_paths = self._get_file_paths()

        # verbose
        if verbose:
            self._print_header()

    def _load_data(self, path, verbose=False):
        """Load data from the specified path"""
        if verbose:
            print(f"Loading data from path: {path}")
        # initial check
        check_path(path)
        # load
        data = torch.load(path, weights_only=False)
        # load  check
        required_keys = [
            "field",
            "condition",
            "field_domain",
            "condition_domain",
            "n_unique_conditions",
            "n_fields_per_sample",
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
                "test": "testset_floral.pt" if self.floral else "testset_flora.pt",
            },
            "statistics": (
                "statistics_floral.pt" if self.floral else "statistics_flora.pt"
            ),
        }
        return file_paths

    def _print_header(self):
        """print the header for the dataloader"""
        print_section("Dataloader config")
        printer(f"Reload datasets: {self.reload}")
        printer(f"Train ratio: {self.train_ratio}")
        printer(f"Val ratio: {self.val_ratio}")
        printer(f"Test ratio: {self.test_ratio}")
        printer(f"Batch size: {self.batch_size}")
        printer(f"Num workers: {self.num_workers}")
        printer(
            f"Normalize field: {self.dataloader_config.normalize.target_field.enabled} "
            f"with Auto: {self.dataloader_config.normalize.target_field.auto}"
        )
        printer(
            "Normalize condition: "
            f"{self.dataloader_config.normalize.condition.enabled} "
            f"with Auto: {self.dataloader_config.normalize.condition.auto}"
        )
        printer(
            "Normalize LF_field: "
            f"{self.dataloader_config.normalize.LF_field.enabled} "
            f"with Auto: {self.dataloader_config.normalize.LF_field.auto}"
        )

        print_section("Dataloader config", end=True)

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

        # type check
        type_check(field, torch.Tensor)
        type_check(condition, torch.Tensor)
        type_check(field_domain, torch.Tensor)
        type_check(condition_domain, torch.Tensor)

        # check size
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
        field_tensor = torch.FloatTensor(field)
        condition_tensor = torch.FloatTensor(condition)
        field_domain_tensor = torch.FloatTensor(field_domain)
        condition_domain_tensor = torch.FloatTensor(condition_domain)

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
        assert (
            HF_condition.shape == HF_field.shape == LF_field.shape
        ), "Low-fidelity field must be defined on the same domain"

        # create target(s)
        if self.floral:
            target_field = HF_field - LF_field
            target_condition = HF_condition
        else:
            target_field = HF_field
            target_condition = HF_condition

        # create operator data dict (keep all realizations; split handles subsetting)
        op_data_dict = {}
        op_data_dict["target_field"] = target_field
        op_data_dict["condition"] = target_condition
        op_data_dict["LF_field"] = LF_field

        # create data shape dict
        shape_dict = {}
        shape_dict["field"] = {
            "channels": HF_field.shape[1],
            "dims": list(HF_field.shape[2:]),
            "ndim": len(HF_field.shape[2:]),
        }
        shape_dict["condition"] = {
            "channels": target_condition.shape[1],
            "dims": list(target_condition.shape[2:]),
            "ndim": len(target_condition.shape[2:]),
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

    @property
    def n_unique_train(self):
        if self.n_train_samples is not None:
            # n_train_samples is total rows; convert to unique conditions
            return int(self.n_train_samples) // self.n_fields_per_sample
        return int(self.n_unique_conditions * self.train_ratio)

    @property
    def n_unique_val(self):
        return int(self.n_unique_conditions * self.val_ratio)

    @property
    def n_unique_test(self):
        return self.n_unique_conditions - self.n_unique_train - self.n_unique_val

    @property
    def n_train(self):
        return self.n_unique_train * self.n_fields_per_sample

    @property
    def n_val(self):
        return self.n_unique_val * self.n_fields_per_sample

    def _condition_rows(self, condition_indices):
        """Return all row indices for the given unique-condition indices.

        Tile layout: condition i lives at rows i, i+n_unique, i+2*n_unique, ...
        """
        n_unique = self.n_unique_conditions
        n_fields = self.n_fields_per_sample
        cond_idx = torch.tensor(list(condition_indices), dtype=torch.long)
        idx = torch.cat([cond_idx + k * n_unique for k in range(n_fields)])
        return idx.sort().values

    def _get_split_data_dict(self, op_data_dict):
        """Split train/val/test by unique input condition to prevent leakage.

        Val and test conditions are anchored to the END of the unique-condition
        pool so they remain stable when n_train_samples is varied.
        """
        n_unique = self.n_unique_conditions
        n_unique_train = self.n_unique_train
        n_unique_val = self.n_unique_val

        # unique-condition index ranges (val/test anchored to end)
        test_cond_start = n_unique - self.n_unique_test
        val_cond_start = test_cond_start - n_unique_val

        assert n_unique_train <= val_cond_start, (
            f"n_unique_train ({n_unique_train}) would overlap with val/test "
            f"(val starts at condition {val_cond_start}). "
            "Reduce n_train_samples or adjust split ratios."
        )

        train_rows = self._condition_rows(range(0, n_unique_train))
        val_rows = self._condition_rows(range(val_cond_start, test_cond_start))
        test_rows = self._condition_rows(range(test_cond_start, n_unique))

        def _index_value(value, rows):
            if isinstance(value, Mapping):
                return {k: _index_value(v, rows) for k, v in value.items()}
            return value[rows]

        train_data_dict = {
            k: _index_value(v, train_rows) for k, v in op_data_dict.items()
        }
        val_data_dict = {k: _index_value(v, val_rows) for k, v in op_data_dict.items()}
        test_data_dict = {
            k: _index_value(v, test_rows) for k, v in op_data_dict.items()
        }
        return train_data_dict, val_data_dict, test_data_dict

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

    def _get_normalize_data_dict(self, train_data_dict, val_data_dict, test_data_dict):
        """Normalize train/val/test using statistics computed from the training data."""
        normalize_keys = ["target_field", "condition", "LF_field"]
        splits = {
            "train": train_data_dict,
            "val": val_data_dict,
            "test": test_data_dict,
        }

        for split_name, split_dict in splits.items():
            check_keys(split_dict, normalize_keys)

        statistics = {}
        normalized = {k: {} for k in splits}

        for key in normalize_keys:
            normalize_config = deep_get(self.dataloader_config, ["normalize", key])
            mean, std = self._get_statistics(
                train_data_dict[key], normalize_config=normalize_config
            )
            statistics[key] = {"mean": mean, "std": std}
            for split_name, split_dict in splits.items():
                normalized[split_name][key] = (split_dict[key] - mean) / std
                check_tensor_blowup(
                    normalized[split_name][key], f"{key} ({split_name})"
                )

        return normalized["train"], normalized["val"], normalized["test"], statistics

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

    def denormalize_LF_field(self, normal_field: torch.Tensor):
        """denormalize the field"""
        # extract field statistics
        field_mean = deep_get(self.statistics, ["LF_field", "mean"])
        field_std = deep_get(self.statistics, ["LF_field", "std"])
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
            train_data_dict, val_data_dict, test_data_dict = self._get_split_data_dict(
                op_data_dict
            )

            # normalize
            (
                train_norm_data_dict,
                val_norm_data_dict,
                test_norm_data_dict,
                statistics,
            ) = self._get_normalize_data_dict(
                train_data_dict=train_data_dict,
                val_data_dict=val_data_dict,
                test_data_dict=test_data_dict,
            )

            # set statistics attribute
            self._set_attribute("statistics", statistics)

            # dataset
            self.train_set = MultiFidelityDataLoader(train_norm_data_dict)
            self.val_set = MultiFidelityDataLoader(val_norm_data_dict)
            self.test_set = MultiFidelityDataLoader(test_norm_data_dict)

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

    def test_dataloader(self):
        """Return the test dataloader"""
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_config.num_workers,
            pin_memory=True,
        )
