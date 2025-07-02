import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.utils.data import DataLoader, TensorDataset
from mfFlow.utils import check_path, get_logger


def t2n(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a PyTorch tensor to a NumPy array."""
    assert isinstance(tensor, torch.Tensor), "Input must be a PyTorch tensor"
    return tensor.detach().cpu().numpy()


def n2t(array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor."""
    assert isinstance(array, np.ndarray), "Input must be a NumPy array"
    return torch.FloatTensor(array)


def get_trainer(checkpointer, logger_name: str, train_config: dict):
    """Get a PyTorch Lightning Trainer with the specified configuration."""
    # get the logger
    logger = get_logger(logger_name)
    # trainer
    trainer = L.Trainer(
        logger=logger,
        max_epochs=train_config.max_epochs,
        devices=train_config.devices,
        accelerator=train_config.accelerator,
        strategy=train_config.strategy,
        callbacks=[checkpointer],
    )

    return trainer


def init_weights(m):
    """weight initializer for the network"""
    if isinstance(m, nn.Linear):
        # Good for ReLU
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class OpDataModule(L.LightningDataModule):
    def __init__(
        self,
        nx: int,
        nc: int,
        nd: int,
        low_fidelity_path: str,
        high_fidelity_path: str,
        n_samples: int,
        n_sensors: int,
        mfFlow: bool,
        dataloader_config: dict,
    ):
        """Base class for data modules in PyTorch Lightning.
        Args:
            nx (int): Dimensionality of the output features.
            nc (int): Dimensionality of the input features.
            nd (int): Dimensionality of the domain, for e.g., 2 for 2D data
            low_fidelity_path (str): Path to the low fidelity data.
            high_fidelity_path (str): Path to the high fidelity data.
            mfFlow (bool): If True, use residual learning
        """
        super(OpDataModule, self).__init__()
        self.nx = nx  # Dimensionality of the output features
        self.nc = nc  # Dimensionality of the input features
        self.nd = nd  # Dimensionality of the domain (e.g., 2 for 2D data)
        self.n_samples = n_samples  # Number of samples in the dataset
        self.n_sensors = n_sensors  # Number of sensors in the output field
        self.mfFlow = mfFlow  # If True, use residual learning
        self.dataloader_config = dataloader_config  # Configuration for the dataloader
        self.train_ratio = dataloader_config.get(
            "train_ratio"
        )  # Ratio of training data
        self.reload = self.dataloader_config.get(
            "reload"
        )  # If True, reload the dataset
        self.batch_size = self.dataloader_config.get(
            "batch_size"
        )  # Batch size for the dataloader
        self.num_workers = self.dataloader_config.get(
            "num_workers"
        )  # Number of workers for the dataloader
        self.normalize = self.dataloader_config.get("normalize")
        self.LF_data = self._load_data(low_fidelity_path)  # Low fidelity data
        self.HF_data = self._load_data(high_fidelity_path)  # High fidelity data

        self.file_paths = {
            "datasets": {
                "train": "trainset_mfFlow.pt" if self.mfFlow else "trainset.pt",
                "val": "valset_mfFlow.pt" if self.mfFlow else "valset.pt",
            },
            "statistics": "statistics_mfFlow.pt" if self.mfFlow else "statistics.pt",
            "test_config": "test_config_mfFlow.pt" if self.mfFlow else "test_config.pt",
        }

    def _extract_fields(self, data_dict: dict):
        """Extract fields from the data dictionary."""
        field = n2t(data_dict.get("field", None))
        condition = n2t(data_dict.get("condition", None))
        domain = n2t(data_dict.get("domain", None))
        # Check shapes of the domain and conditions
        # field shape will change afterwards
        assert (
            field.shape[1] == self.nc
        ), f"Field shape mismatch: expected {self.nc}, got {field.shape[1]}"
        assert (
            condition.shape[1] == self.nc
        ), f"Condition shape mismatch: expected {self.nc}, got {condition.shape[1]}"
        assert (
            domain.shape[1] == self.nd
        ), f"Domain shape mismatch: expected {self.nd}, got {domain.shape[1]}"

        return field, condition, domain

    def _subselect_samples(self, field: torch.Tensor, condition: torch.Tensor):
        """Sub-select a fixed number of samples from the data."""
        n_samples_available = field.shape[0]
        assert (
            n_samples_available >= self.n_samples
        ), f"Not enough samples available: {n_samples_available} < {self.n_samples}"
        field_sub = field[: self.n_samples, :]
        condition_sub = condition[: self.n_samples, :]
        return field_sub, condition_sub

    def _process_operator_fields(self):
        """Process operator fields for the data module."""
        # Extract fields from low fidelity data
        LF_field, LF_condition, LF_domain = self._extract_fields(self.LF_data)
        # Extract fields from high fidelity data
        HF_field, HF_condition, HF_domain = self._extract_fields(self.HF_data)
        # Assert statements to ensure the shapes are correct
        assert (
            LF_field.shape[0] == HF_field.shape[0]
        ), "Low fidelity and high fidelity fields must have the same number of samples"

        # Sub-select the samples
        (
            LF_field_sub,
            LF_condition_sub,
        ) = self._subselect_samples(LF_field, LF_condition)
        (
            HF_field_sub,
            HF_condition_sub,
        ) = self._subselect_samples(HF_field, HF_condition)

        # Get the sensor locations
        n_sensors_available = LF_field.shape[1]
        assert (
            n_sensors_available >= self.n_sensors
        ), f"Not enough sensors available: {n_sensors_available} < {self.n_sensors}"

        sensor_locations = torch.stack(
            [torch.randperm(self.nc)[: self.n_sensors] for _ in range(self.n_samples)],
            dim=0,
        )

        # Get the field at sensor locations
        LF_field_sensor = LF_field_sub.gather(1, sensor_locations)
        HF_field_sensor = HF_field_sub.gather(1, sensor_locations)

        # Flatten the fields to match the expected shape
        LF_field_flat = LF_field_sensor.view(-1, self.nx)
        HF_field_flat = HF_field_sensor.view(-1, self.nx)

        # Process the domain
        domain_op = HF_domain.unsqueeze(0).repeat(self.n_samples, 1, 1)
        domain_sensor = domain_op.gather(
            1, sensor_locations.unsqueeze(-1).repeat(1, 1, self.nd)
        )
        domain = domain_sensor.view(-1, self.nd)

        # Process the conditions
        condition = (
            HF_condition_sub.unsqueeze(1).repeat(1, self.n_sensors, 1).view(-1, self.nc)
        )

        # Process the field
        if self.mfFlow:
            field = HF_field_flat - LF_field_flat
        else:
            field = HF_field_flat

        # Create the data dict
        data_dict = {}
        data_dict["field"] = field
        data_dict["condition"] = condition
        data_dict["domain"] = domain
        data_dict["sensor_locations"] = sensor_locations
        data_dict["LF_field"] = LF_field_sub
        data_dict["HF_field"] = HF_field_sub
        data_dict["HF_condition"] = HF_condition_sub
        data_dict["LF_condition"] = LF_condition_sub
        data_dict["HF_domain"] = HF_domain
        data_dict["LF_domain"] = LF_domain

        return data_dict

    def _split_data(self, data_dict: dict):
        """Split the data into training and validation sets."""
        # Extract the fields from the data dictionary
        field = data_dict.get("field", None)
        condition = data_dict.get("condition", None)
        domain = data_dict.get("domain", None)

        n_train = int(self.train_ratio * len(field))

        field_train, field_val = field[:n_train], field[n_train:]
        condition_train, condition_val = condition[:n_train], condition[n_train:]
        domain_train, domain_val = domain[:n_train], domain[n_train:]

        train_data = {
            "field": field_train,
            "condition": condition_train,
            "domain": domain_train,
        }
        val_data = {
            "field": field_val,
            "condition": condition_val,
            "domain": domain_val,
        }
        return train_data, val_data

    def _normalize_data(self, train_data: dict, val_data: dict):
        # Normalize the fields
        field_train = train_data["field"]
        field_val = val_data["field"]
        if self.normalize.field:
            field_mean = field_train.mean(dim=0, keepdim=True)
            field_std = field_train.std(dim=0, keepdim=True)
            field_train_norm = (field_train - field_mean) / field_std
            field_val_norm = (field_val - field_mean) / field_std
        else:
            field_mean = torch.zeros(1, self.nx)
            field_std = torch.ones(1, self.nx)
            field_train_norm = field_train
            field_val_norm = field_val
        # Normalize the conditions
        condition_train = train_data["condition"]
        condition_val = val_data["condition"]
        if self.normalize.condition:
            condition_mean = condition_train.mean(dim=0, keepdim=True)
            condition_std = condition_train.std(dim=0, keepdim=True)
            condition_train_norm = (condition_train - condition_mean) / condition_std
            condition_val_norm = (condition_val - condition_mean) / condition_std
        else:
            condition_mean = torch.zeros(1, self.nc)
            condition_std = torch.ones(1, self.nc)
            condition_train_norm = condition_train
            condition_val_norm = condition_val
        # Check NaNs and Infs
        assert not torch.isnan(
            field_train_norm
        ).any(), "NaN values found in field_train_norm"
        assert not torch.isnan(
            field_val_norm
        ).any(), "NaN values found in field_val_norm"
        assert not torch.isnan(
            condition_train_norm
        ).any(), "NaN values found in condition_train_norm"
        assert not torch.isnan(
            condition_val_norm
        ).any(), "NaN values found in condition_val_norm"
        # Data dict
        train_data_norm = {
            "field": field_train_norm,
            "condition": condition_train_norm,
            "domain": train_data["domain"],
        }
        val_data_norm = {
            "field": field_val_norm,
            "condition": condition_val_norm,
            "domain": val_data["domain"],
        }

        # Statistics dict
        self.statistics = {
            "field": {"mean": field_mean, "std": field_std},
            "condition": {"mean": condition_mean, "std": condition_std},
        }

        return train_data_norm, val_data_norm

    def _check_keys(self, data):
        """Check if the data has the required keys."""
        required_keys = ["field", "condition", "domain"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Data must contain the key '{key}'")

    def _load_data(self, path: str):
        """Load data from the specified path."""
        check_path(path)
        data = np.load(path, allow_pickle=True)

        # check if the data has the required keys
        self._check_keys(data)

        return data

    def _get_test_config(self, data_dict: dict):
        """prepare the test config
        Notes:
        - Only condition is normalized as the fields are used for only comparison.
        """
        LF_field = data_dict.get("LF_field", None)
        HF_field = data_dict.get("HF_field", None)
        HF_condition = data_dict.get("HF_condition", None)
        HF_domain = data_dict.get("HF_domain", None)

        # Normalize the condition
        condition_mean = self.statistics["condition"]["mean"]
        condition_std = self.statistics["condition"]["std"]
        condition = (HF_condition - condition_mean) / condition_std

        # Create config dict
        test_config = {
            "LF_field": LF_field,
            "HF_field": HF_field,
            "condition": condition,
            "domain": HF_domain,
        }

        return test_config

    def setup(self, stage: str = None):
        if self.reload:
            raise NotImplementedError("Reloading the dataset is not implemented yet")
        else:
            # Get the processed operator fields
            op_data_dict = self._process_operator_fields()

            # Split the data into training and validation sets
            train_data, val_data = self._split_data(op_data_dict)

            # Normalize the data and setattr `statistics`
            train_data_norm, val_data_norm = self._normalize_data(train_data, val_data)

            # Prepare test configuration
            self.test_config = self._get_test_config(op_data_dict)

            # Create datasets
            self.train_set = TensorDataset(
                train_data_norm["field"],
                train_data_norm["condition"],
                train_data_norm["domain"],
            )

            self.val_set = TensorDataset(
                val_data_norm["field"],
                val_data_norm["condition"],
                val_data_norm["domain"],
            )
            # Save
            torch.save(self.train_set, self.file_paths["datasets"]["train"])
            torch.save(self.val_set, self.file_paths["datasets"]["val"])
            torch.save(self.statistics, self.file_paths["statistics"])
            torch.save(self.test_config, self.file_paths["test_config"])

    def train_dataloader(self):
        """Returns the training dataloader."""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
