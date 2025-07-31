import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from mfFlow.utils import check_path, get_logger, printer


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
    assert isinstance(train_config, dict), "train_config must be a dictionary."
    # get the logger
    logger = get_logger(logger_name)

    # early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        min_delta=1e-4,  # Minimum change to qualify as improvement
        patience=int(
            0.2 * train_config["max_epochs"]
        ),  # Number of epochs with no improvement after which training will stop
        verbose=True,
        mode="min",  # "min" for loss, "max" for accuracy
    )

    # trainer
    trainer = L.Trainer(
        logger=logger,
        max_epochs=train_config["max_epochs"],
        devices=train_config["devices"],
        accelerator=train_config["accelerator"],
        strategy=train_config["strategy"],
        callbacks=[checkpointer, early_stop_callback],
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
        sensing_strategy: str,
        dataloader_config: dict,
        test_data_path: str,
    ):
        """Base class for data modules in PyTorch Lightning.
        Args:
            nx (int): Dimensionality of the output features.
            nc (int): Dimensionality of the input features.
            nd (int): Dimensionality of the domain, for e.g., 2 for 2D data
            low_fidelity_path (str): Path to the low fidelity data.
            high_fidelity_path (str): Path to the high fidelity data.
            mfFlow (bool): If True, use residual learning
            sensing_strategy (str): Strategy for selecting sensor locations.
            dataloader_config (dict): Configuration for the dataloader.
            test_data_path (str): Path for the test data.
        """
        super(OpDataModule, self).__init__()
        self.nx = nx  # Dimensionality of the output features
        self.nc = nc  # Dimensionality of the input features
        self.nd = nd  # Dimensionality of the domain (e.g., 2 for 2D data)
        self.n_samples = n_samples  # Number of samples in the dataset
        self.n_sensors = n_sensors  # Number of sensors in the output field
        self.mfFlow = mfFlow  # If True, use residual learning
        self.sensing_strategy = (
            sensing_strategy.lower().strip()
        )  # Strategy for selecting sensor locations
        self.dataloader_config = dataloader_config  # Configuration for the dataloader
        assert isinstance(
            self.dataloader_config, dict
        ), "dataloader_config must be a dictionary."
        self.test_data_path = test_data_path  # Path for the test data
        self.train_ratio = self.dataloader_config.get(
            "train_ratio"
        )  # Ratio of training data
        self.reload = self.dataloader_config.get(
            "reload"
        )  # If True, reload the dataset

        # if True, reload the sensors
        # Note, if self.reload is True, self.reload_sensors will be ignored and
        # the sensors will be reloaded
        self.reload_sensors = self.dataloader_config.get("reload_sensors")

        self.batch_size = self.dataloader_config[
            "batch_size"
        ]  # Batch size for the dataloader
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
            "sensor_locations": "sensor_locations_mfFlow.pt"
            if self.mfFlow
            else "sensor_locations.pt",
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

    def _get_sensor_locations(self):
        """Get the sensor locations based on the sensing strategy.
        Returns:
            sensor_locations (torch.Tensor): Tensor of shape (n_samples, n_sensors)
        """
        if self.reload_sensors:
            raise NotImplementedError("Pass sensor file instead from config.")
            printer(
                f"Reloading sensor locations from {self.file_paths['sensor_locations']}"
            )
            check_path(self.file_paths["sensor_locations"], "to disable reload_sensors")
            sensor_locations = torch.load(self.file_paths["sensor_locations"])
            # Check shape
            assert sensor_locations.shape == (self.n_samples, self.n_sensors), (
                "Sensor locations shape mismatch:"
                + f" Expected {self.n_samples, self.n_sensors}, "
                + f"got {sensor_locations.shape}"
            )
        else:
            sensing_strategies = ["random", "uniform", "stratified"]
            assert self.sensing_strategy in sensing_strategies, (
                f"Invalid sensing strategy: {self.sensing_strategy}."
                f"Choose from {','.join(sensing_strategies)} "
            )

            if self.sensing_strategy == "random":
                # randomly select sensors from the available sensors
                sensor_locations = torch.stack(
                    [
                        torch.randperm(self.nc)[: self.n_sensors]
                        for _ in range(self.n_samples)
                    ],
                    dim=0,
                )
            elif self.sensing_strategy == "uniform":
                # Uniformly select sensors across the available sensors
                sensor_locations = torch.linspace(
                    0, self.nc - 1, self.n_sensors, dtype=torch.long
                ).repeat(self.n_samples, 1)

            elif self.sensing_strategy == "stratified":
                # Divide the domain into equal parts and select sensors from each part
                n_bins = 3  # divide the domain into n_bins parts
                bins = torch.linspace(0, self.nc, n_bins + 1, dtype=torch.long)
                sensors_per_bin = self.n_sensors // n_bins
                remainder = self.n_sensors % n_bins

                sensors_per_bin_list = [sensors_per_bin] * n_bins
                sensors_per_bin_list[:remainder] = [
                    x + 1 for x in sensors_per_bin_list[:remainder]
                ]

                sensor_locations = torch.stack(
                    [
                        torch.hstack(
                            [
                                torch.randint(
                                    bins[ii].item(),
                                    bins[ii + 1].item(),
                                    (sensors_per_bin_list[ii],),
                                )
                                for ii in range(n_bins)
                            ]
                        )
                        for jj in range(self.n_samples)
                    ]
                )

            else:
                raise ValueError(f"Invalid sensing strategy: {self.sensing_strategy}")
        assert sensor_locations.shape == (
            self.n_samples,
            self.n_sensors,
        ), "invalid sensor selection"
        assert (
            sensor_locations.max() < self.nc
        ), f"sensor out of bounds: {sensor_locations.max()} >= {self.nc}"

        return sensor_locations

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

        sensor_locations = self._get_sensor_locations()

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

            # Avoid division by small values
            field_std = torch.where(
                field_std < 1e-10, torch.ones_like(field_std), field_std
            )

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

            # Avoid division by small values
            condition_std = torch.where(
                condition_std < 1e-10, torch.ones_like(condition_std), condition_std
            )

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

    def _get_test_config(self):
        """prepare the test config
        Notes:
        - Only condition is normalized as the fields are used for only comparison.
        """
        # Check if test config is avilable
        check_path(self.test_data_path)
        # Load the test data
        test_data = np.load(self.test_data_path, allow_pickle=True)
        # Check keys
        req_keys = ["LF_field", "HF_field", "condition", "domain"]
        assert all(
            test_key in test_data for test_key in req_keys
        ), f"Test data must contain the keys: {req_keys}"

        LF_field = test_data.get("LF_field", None)
        HF_field = test_data.get("HF_field", None)
        condition = test_data.get("condition", None)
        domain = test_data.get("domain", None)

        # Check shapes
        assert (
            LF_field.shape[1] == self.nc
        ), f"LF_field shape mismatch: expected {self.nc}, got {LF_field.shape[1]}"
        assert (
            HF_field.shape[1] == self.nc
        ), f"HF_field shape mismatch: expected {self.nc}, got {HF_field.shape[1]}"
        assert (
            condition.shape[1] == self.nc
        ), f"Condition shape mismatch: expected {self.nc}, got {condition.shape[1]}"
        assert (
            domain.shape[1] == self.nd
        ), f"Domain shape mismatch: expected {self.nd}, got {domain.shape[1]}"

        # Convert to tensors
        LF_field = n2t(LF_field)
        HF_field = n2t(HF_field)
        condition = n2t(condition)
        domain = n2t(domain)

        # Normalize the condition
        condition_mean = self.statistics["condition"]["mean"]
        condition_std = self.statistics["condition"]["std"]
        condition = (condition - condition_mean) / condition_std

        assert not torch.isnan(condition).any(), "Invalid normalization"

        # Create config dict
        test_config = {
            "LF_field": LF_field,
            "HF_field": HF_field,
            "condition": condition,
            "domain": domain,
            "n_samples": len(condition),
        }

        return test_config

    def setup(self, stage: str = None):
        if self.reload:
            # Load the datasets if reload is True
            check_path(self.file_paths["datasets"]["train"])
            check_path(self.file_paths["datasets"]["val"])
            check_path(self.file_paths["statistics"])
            check_path(self.file_paths["test_config"])
            check_path(self.file_paths["sensor_locations"])

            self.train_set = torch.load(
                self.file_paths["datasets"]["train"], weights_only=False
            )
            self.val_set = torch.load(
                self.file_paths["datasets"]["val"], weights_only=False
            )
            self.statistics = torch.load(
                self.file_paths["statistics"], weights_only=False
            )
            self.test_config = torch.load(
                self.file_paths["test_config"], weights_only=False
            )

            # Load sensor locations irrespective of self.reload_sensors
            self.sensor_locations = torch.load(
                self.file_paths["sensor_locations"], weights_only=False
            )

        else:
            # Get the processed operator fields
            op_data_dict = self._process_operator_fields()

            # Sensor locations
            self.sensor_locations = op_data_dict.get("sensor_locations", None)

            # Split the data into training and validation sets
            train_data, val_data = self._split_data(op_data_dict)

            # Normalize the data and setattr `statistics`
            train_data_norm, val_data_norm = self._normalize_data(train_data, val_data)

            # Prepare test configuration
            self.test_config = self._get_test_config()

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
            torch.save(self.sensor_locations, self.file_paths["sensor_locations"])

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


class GPDataModule:
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
        test_data_path: str,
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
        super(GPDataModule, self).__init__()
        self.nx = nx  # Dimensionality of the output features
        self.nc = nc  # Dimensionality of the input features
        self.nd = nd  # Dimensionality of the domain (e.g., 2 for 2D data)
        self.n_samples = n_samples  # Number of samples in the dataset
        self.n_sensors = n_sensors  # Number of sensors in the output field
        self.mfFlow = mfFlow  # If True, use residual learning
        self.dataloader_config = dataloader_config  # Configuration for the dataloader
        self.test_data_path = test_data_path  # Path for the test data
        self.train_ratio = self.dataloader_config.get(
            "train_ratio"
        )  # Ratio of training data
        self.reload = self.dataloader_config.get(
            "reload"
        )  # If True, reload the dataset

        # if True, reload the sensors
        # Note, if self.reload is True, self.reload_sensors will be ignored and
        # the sensors will be reloaded
        self.reload_sensors = self.dataloader_config.get("reload_sensors")

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
                "train": "trainset_mfFlow_GP.pt" if self.mfFlow else "trainset_GP.pt",
                "val": "valset_mfFlow_GP.pt" if self.mfFlow else "valset_GP.pt",
            },
            "statistics": "statistics_mfFlow_GP.pt"
            if self.mfFlow
            else "statistics_GP.pt",
            "test_config": "test_config_mfFlow_GP.pt"
            if self.mfFlow
            else "test_config_GP.pt",
            "sensor_locations": "sensor_locations_mfFlow_GP.pt"
            if self.mfFlow
            else "sensor_locations_GP.pt",
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

    def _process_gp_fields(self):
        """Process gaussian process fields for the data module."""
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
        if self.reload_sensors:
            raise NotImplementedError("Load sensor form file. pass in config.")
            printer(
                f"Reloading sensor locations from {self.file_paths['sensor_locations']}"
            )
            check_path(self.file_paths["sensor_locations"], "to disable reload_sensors")
            sensor_locations = torch.load(self.file_paths["sensor_locations"])
            # Check shape
            assert sensor_locations.shape == (self.n_samples, self.n_sensors), (
                "Sensor locations shape mismatch:"
                + f" Expected {self.n_samples, self.n_sensors}, "
                + f"got {sensor_locations.shape}"
            )
        else:
            sensor_locations = torch.stack(
                [
                    torch.randperm(self.nc)[: self.n_sensors]
                    for _ in range(self.n_samples)
                ],
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
        condition = HF_condition_sub.gather(1, sensor_locations).ravel().unsqueeze(1)
        # Process the field
        if self.mfFlow:
            field = HF_field_flat - LF_field_flat
        else:
            field = HF_field_flat
        # Create the in_features (condition + domain)
        in_features = torch.cat([condition, domain], dim=1)
        # Create the output features (field)
        out_features = field

        # Create the data dict
        data_dict = {}
        data_dict["in_features"] = in_features  # input features to the GP
        data_dict["out_features"] = out_features  # output features from the GP
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
        # Extract the in_features
        in_features = data_dict.get("in_features", None)
        # Extract the out_features
        out_features = data_dict.get("out_features", None)

        n_train = int(self.train_ratio * len(in_features))

        # split
        in_features_train, in_features_val = (
            in_features[:n_train],
            in_features[n_train:],
        )
        out_features_train, out_features_val = (
            out_features[:n_train],
            out_features[n_train:],
        )

        train_data = {
            "in_features": in_features_train,
            "out_features": out_features_train,
        }
        val_data = {
            "in_features": in_features_val,
            "out_features": out_features_val,
        }
        return train_data, val_data

    def _normalize_data(self, train_data: dict, val_data: dict):
        # In features
        in_features_train = train_data["in_features"]
        in_features_val = val_data["in_features"]
        if self.normalize.condition:
            in_features_mean = in_features_train.mean(dim=0, keepdim=True)
            in_features_std = in_features_train.std(dim=0, keepdim=True)
            in_features_train_norm = (
                in_features_train - in_features_mean
            ) / in_features_std
            in_features_val_norm = (
                in_features_val - in_features_mean
            ) / in_features_std
        else:
            in_features_mean = torch.zeros(1, self.nc + self.nd)
            in_features_std = torch.ones(1, self.nc + self.nd)
            in_features_train_norm = in_features_train
            in_features_val_norm = in_features_val

        # Out features
        out_features_train = train_data["out_features"]
        out_features_val = val_data["out_features"]
        if self.normalize.field:
            out_features_mean = out_features_train.mean(dim=0, keepdim=True)
            out_features_std = out_features_train.std(dim=0, keepdim=True)
            out_features_train_norm = (
                out_features_train - out_features_mean
            ) / out_features_std
            out_features_val_norm = (
                out_features_val - out_features_mean
            ) / out_features_std
        else:
            out_features_mean = torch.zeros(1, self.nx)
            out_features_std = torch.ones(1, self.nx)
            out_features_train_norm = out_features_train
            out_features_val_norm = out_features_val

        # Check NaNs and Infs
        assert not torch.isnan(
            in_features_train_norm
        ).any(), "NaN values found in in_features_train_norm"
        assert not torch.isnan(
            in_features_val_norm
        ).any(), "NaN values found in in_features_val_norm"
        assert not torch.isnan(
            out_features_train_norm
        ).any(), "NaN values found in out_features_train_norm"
        assert not torch.isnan(
            out_features_val_norm
        ).any(), "NaN values found in out_features_val_norm"
        # Data dict
        train_data_norm = {
            "in_features": in_features_train_norm,
            "out_features": out_features_train_norm,
        }
        val_data_norm = {
            "in_features": in_features_val_norm,
            "out_features": out_features_val_norm,
        }

        # Statistics dict
        self.statistics = {
            "in_features": {"mean": in_features_mean, "std": in_features_std},
            "out_features": {"mean": out_features_mean, "std": out_features_std},
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

    def _get_test_config(self):
        """prepare the test config
        Notes:
        - Only condition is normalized as the fields are used for only comparison.
        """
        # Check if test config is avilable
        check_path(self.test_data_path)
        # Load the test data
        test_data = np.load(self.test_data_path, allow_pickle=True)
        # Check keys
        req_keys = ["LF_field", "HF_field", "condition", "domain"]
        assert all(
            test_key in test_data for test_key in req_keys
        ), f"Test data must contain the keys: {req_keys}"
        LF_field = test_data.get("LF_field", None)
        HF_field = test_data.get("HF_field", None)
        condition = test_data.get("condition", None)
        domain = test_data.get("domain", None)
        # Check shapes
        assert (
            LF_field.shape[1] == self.nc
        ), f"LF_field shape mismatch: expected {self.nc}, got {LF_field.shape[1]}"
        assert (
            HF_field.shape[1] == self.nc
        ), f"HF_field shape mismatch: expected {self.nc}, got {HF_field.shape[1]}"
        assert (
            condition.shape[1] == self.nc
        ), f"Condition shape mismatch: expected {self.nc}, got {condition.shape[1]}"
        assert (
            domain.shape[1] == self.nd
        ), f"Domain shape mismatch: expected {self.nd}, got {domain.shape[1]}"

        # Convert to tensors
        LF_field = n2t(LF_field)
        HF_field = n2t(HF_field)
        condition = n2t(condition)
        domain = n2t(domain)

        # Normalize the in_features (condition + domain)
        domain_batch = (
            domain.unsqueeze(0).repeat(len(condition), 1, 1).view(-1, self.nd)
        )
        condition_batch = condition.view(-1, 1)
        assert len(domain_batch) == len(condition_batch), "Incorrect shape."
        in_features = torch.cat([condition_batch, domain_batch], dim=1)
        in_features_mean = self.statistics["in_features"]["mean"]
        in_features_std = self.statistics["in_features"]["std"]
        in_features = (in_features - in_features_mean) / in_features_std

        # Create config dict
        test_config = {
            "in_features": in_features,
            "LF_field": LF_field,
            "HF_field": HF_field,
            "condition": condition,
            "domain": domain,
            "n_samples": len(condition),
        }

        return test_config

    def setup(self, stage: str = None):
        if self.reload:
            raise NotImplementedError("Under construction. Please wait...")
            # Load the datasets if reload is True
            check_path(self.file_paths["datasets"]["train"])
            check_path(self.file_paths["datasets"]["val"])
            check_path(self.file_paths["statistics"])
            check_path(self.file_paths["test_config"])
            self.train_set = torch.load(self.file_paths["datasets"]["train"])
            self.val_set = torch.load(self.file_paths["datasets"]["val"])
            self.statistics = torch.load(self.file_paths["statistics"])
            self.test_config = torch.load(self.file_paths["test_config"])

            # Load sensor locations irrespective of self.reload_sensors
            self.sensor_locations = torch.load(self.file_paths["sensor_locations"])

        else:
            # Get the processed operator fields
            gp_data_dict = self._process_gp_fields()

            # Sensor locations
            self.sensor_locations = gp_data_dict.get("sensor_locations", None)

            # Split the data into training and validation sets
            train_data, val_data = self._split_data(gp_data_dict)

            # Normalize the data and setattr `statistics`
            train_data_norm, val_data_norm = self._normalize_data(train_data, val_data)

            # Prepare test configuration
            self.test_config = self._get_test_config()

            # Create datasets
            self.train_set = TensorDataset(
                train_data_norm["in_features"],
                train_data_norm["out_features"],
            )

            self.val_set = TensorDataset(
                val_data_norm["in_features"],
                val_data_norm["out_features"],
            )
            # Save
            torch.save(self.train_set, self.file_paths["datasets"]["train"])
            torch.save(self.val_set, self.file_paths["datasets"]["val"])
            torch.save(self.statistics, self.file_paths["statistics"])
            torch.save(self.test_config, self.file_paths["test_config"])
            torch.save(self.sensor_locations, self.file_paths["sensor_locations"])


class RunningAverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self._reset()

    def _reset(self):
        self.current_val = None
        self.current_avg = None
        self.val_history = []
        self.avg_history = []
        self.best_val = None

    def update(self, val):
        if self.current_val is None:
            self.current_val = val
            self.current_avg = val
            self.best_val = self.current_val
        else:
            self.current_val = val
            self.current_avg = (
                self.momentum * self.current_avg + (1 - self.momentum) * val
            )
            self.best_val = min(self.best_val, self.current_val)
        self.val_history.append(self.current_val)
        self.avg_history.append(self.current_avg)
