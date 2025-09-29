# src/floral/utils/utils_train.py
"""
Utilities for model training
"""
import numpy as np
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from .utils_IO import check_path, get_logger, printer


def t2n(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a PyTorch tensor to a NumPy array."""
    assert isinstance(tensor, torch.Tensor), "Input must be a PyTorch tensor"
    return tensor.detach().cpu().numpy()


def n2t(array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor."""
    assert isinstance(array, np.ndarray), "Input must be a NumPy array"
    return torch.FloatTensor(array)


def get_trainer(
    checkpointer, logger_name: str, train_config: dict, verbose: bool = False
):
    """Get a PyTorch Lightning Trainer with the specified configuration."""
    assert isinstance(train_config, dict), "train_config must be a dictionary."
    # get the logger
    logger = get_logger(logger_name)

    # extract train config
    accelerator = train_config.get("accelerator", "cpu")
    devices = train_config.get("devices", 1)
    precision = train_config.get("precision", "bf16-mixed")
    max_epochs = train_config.get("max_epochs", 100)

    # assert
    assert precision == "32-true", (
        "Model currently uses FNO and does not account for cuFFT pow(2)"
        "requirement for half precision"
    )

    # early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        min_delta=1e-4,  # Minimum change to qualify as improvement
        patience=int(
            0.2 * max_epochs
        ),  # Number of epochs with no improvement after which training will stop
        verbose=True,
        mode="min",  # "min" for loss, "max" for accuracy
    )

    # trainer
    trainer = L.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        devices=devices,  # currently FNO supports single device
        accelerator=accelerator,
        precision=precision,
        callbacks=[checkpointer, early_stop_callback],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    if verbose:
        printer("==" * 50)
        printer("**" * 10 + "Trainer config" + "**" * 10)
        printer(f"Logger file: {logger_name}")
        printer(f"Running on : {accelerator}")
        printer(f"Number of devices: {devices}")
        printer(f"Precision: {precision}")
        printer(f"Max epochs: {max_epochs}")
        printer("==" * 50)

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
    """Data module for operator learning.
    Attributes:
        low_fidelity_path (str): Path to the low fidelity data.
        high_fidelity_path (str): Path to the high fidelity data.
        n_samples (int): Number of samples in the dataset (Train + Val).
        This is only done to train model in memory constrained environments.
        dataloader_config (dict): Configuration for the dataloader.
        floral (bool): If True, use residual learning.
    """

    def __init__(
        self,
        low_fidelity_path: str,
        high_fidelity_path: str,
        n_samples: int,
        dataloader_config: dict,
        floral: bool = False,
        **kwargs,
    ):
        super(OpDataModule, self).__init__()
        self.low_fidelity_path = low_fidelity_path
        self.high_fidelity_path = high_fidelity_path
        self.n_samples = n_samples
        self.dataloader_config = dataloader_config
        self.floral = floral
        self.verbose = kwargs.get("verbose", False)

        # extract hte dataloader config
        self.reload = self.dataloader_config.get("reload", False)
        self.train_ratio = self.dataloader_config.get("train_ratio", 0.8)
        self.batch_size = self.dataloader_config.get("batch_size", 32)
        self.num_workers = self.dataloader_config.get("num_workers", 4)
        self.normalize = self.dataloader_config.get("normalize", None)

        # load the data
        self.LF_data = self._load_data(path=self.low_fidelity_path)
        self.HF_data = self._load_data(path=self.high_fidelity_path)

        # get file paths
        self.file_paths = self._get_file_paths()

        # assertions
        assert isinstance(
            self.dataloader_config, dict
        ), "dataloader_config must be a dictionary."

        if self.verbose:
            self._print_header()

    def _print_header(self):
        """print the header for the dataloader"""
        printer("==" * 50)
        printer("**" * 10 + "Dataloader config" + "**" * 10)
        printer(f"Reload datasets: {self.reload}")
        printer(f"Train/Val ratio: {self.train_ratio}")
        printer(f"Batch size: {self.batch_size}")
        printer(f"Num workers: {self.num_workers}")
        printer(
            f"Normalize field: {self.normalize.field.enabled}"
            f"with Auto: {self.normalize.field.auto}"
        )
        printer(
            f"Normalize condition: {self.normalize.condition.enabled}"
            f"with Auto: {self.normalize.condition.auto}"
        )
        printer("==" * 50)

    def _get_file_paths(self):
        """Get the file paths for saving/loading datasets and statistics"""
        # file paths
        file_paths = {
            "datasets": {
                "train": "trainset_floral.pt" if self.floral else "trainset.pt",
                "val": "valset_floral.pt" if self.floral else "valset.pt",
            },
            "statistics": "statistics_floral.pt" if self.floral else "statistics.pt",
            "domains": "domains_floral.pt" if self.floral else "domains.pt",
        }
        return file_paths

    def _load_data(self, path: str):
        """Load data from the specified path"""
        check_path(path)
        data = np.load(path, allow_pickle=True)
        self._check_required_data_keys(data)
        return data

    def _check_required_data_keys(self, data):
        """Check if the data has the required keys"""
        required_keys = [
            "field",
            "field_domain",
            "condition",
            "condition_domain",
        ]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Data must contain the key '{key}'")

    def _extract_fields(self, data_dict: dict):
        field = n2t(data_dict.get("field", None))
        field_domain = n2t(data_dict.get("field_domain", None))

        condition = n2t(data_dict.get("condition", None))
        condition_domain = n2t(data_dict.get("condition_domain", None))

        # extract shapes
        _, fields_ch, *field_grid = field.shape
        _, condition_ch, *condition_grid = condition.shape

        # assert statements
        assert (
            field.shape[0] == condition.shape[0]
        ), "Number of samples in field and condition must be the same."

        assert field_domain.ndim == 2, "Field domain must be 2D (flattned)."
        assert condition_domain.ndim == 2, "Condition domain must be 2D (flattned)."

        assert (
            len(field_grid) == field_domain.shape[1]
        ), "Field and field_domain dimensions must match."
        assert (
            len(condition_grid) == condition_domain.shape[1]
        ), "Condition and condition_domain dimensions must match."

        assert field_domain.shape[0] == int(
            np.prod(field_grid)
        ), "Field domain shape mismatch."
        assert condition_domain.shape[0] == int(
            np.prod(condition_grid)
        ), "Condition domain shape mismatch."

        assert isinstance(
            field, torch.FloatTensor
        ), "Field must be a torch.FloatTensor."
        assert isinstance(
            field_domain, torch.FloatTensor
        ), "Field domain must be a torch.FloatTensor."
        assert isinstance(
            condition, torch.FloatTensor
        ), "Condition must be a torch.FloatTensor."
        assert isinstance(
            condition_domain, torch.FloatTensor
        ), "Condition domain must be a torch.FloatTensor."

        return field, field_domain, condition, condition_domain

    def _subselect_samples(self, field, condition):
        assert len(field) >= self.n_samples, "Not enough samples in the data."
        field_sub = field[: self.n_samples]
        condition_sub = condition[: self.n_samples]
        return field_sub, condition_sub

    def _process_operator_fields(self):
        """
        process the operator fields by subselecting samples (for tractability)
        and applying floral if needed
        """
        # extract the fields
        (
            LF_field,
            LF_field_domain,
            LF_condition,
            LF_condition_domain,
        ) = self._extract_fields(self.LF_data)
        (
            HF_field,
            HF_field_domain,
            HF_condition,
            HF_condition_domain,
        ) = self._extract_fields(self.HF_data)

        # subselect samples
        LF_field_sub, LF_condition_sub = self._subselect_samples(
            field=LF_field, condition=LF_condition
        )

        HF_field_sub, HF_condition_sub = self._subselect_samples(
            field=HF_field, condition=HF_condition
        )

        # check field shape (low fidelity MUST be interpolated to the high fidelity dim)
        assert LF_field_sub.shape == HF_field_sub.shape, "incompatible field shapes"
        assert (
            LF_condition_sub.shape == HF_condition_sub.shape
        ), "incompatible cond shapes"

        if self.floral:
            field = HF_field_sub - LF_field_sub
        else:
            field = HF_field_sub

        # create the data dict
        data_dict = {}
        data_dict["field"] = field
        data_dict["field_domain"] = HF_field_domain
        data_dict["condition"] = HF_condition_sub
        data_dict["condition_domain"] = HF_condition_domain
        data_dict[
            "LF_field"
        ] = LF_field_sub  # to add back the low fidelity field during inference

        return data_dict

    def _split_data(self, data_dict: dict):
        """Split the data into training and validation sets"""
        # extract the fields
        field = data_dict.get("field", None)
        field_domain = data_dict.get("field_domain", None)
        condition = data_dict.get("condition", None)
        condition_domain = data_dict.get("condition_domain", None)
        LF_field = data_dict.get("LF_field", None)

        n_train = int(self.train_ratio * len(field))

        field_train, field_val = field[:n_train], field[n_train:]
        condition_train, condition_val = condition[:n_train], condition[n_train:]
        LF_field_train, LF_field_val = LF_field[:n_train], LF_field[n_train:]

        # create the data dicts
        train_data = {
            "field": field_train,
            "field_domain": field_domain,
            "condition": condition_train,
            "condition_domain": condition_domain,
            "LF_field": LF_field_train,  # ONLY to add back during infer. (=True)
        }

        val_data = {
            "field": field_val,
            "field_domain": field_domain,
            "condition": condition_val,
            "condition_domain": condition_domain,
            "LF_field": LF_field_val,  # ONLY to add back during infer. (=True)
        }

        return train_data, val_data

    def _normalize_field(self, field_train: torch.Tensor, field_val: torch.Tensor):
        """Normalize the field"""
        _, field_ch, *field_grid = field_train.shape

        if self.normalize.field.enabled:
            if self.normalize.field.auto:
                # compute mean and std
                field_mean = field_train.mean(
                    dim=(0, *range(2, field_train.ndim)), keepdim=True
                )  # mean per channel
                field_std = field_train.std(
                    dim=(0, *range(2, field_train.ndim)), keepdim=True
                )
            else:
                field_mean = self.normalize.field.mean
                field_std = self.normalize.field.std
                assert field_mean is not None, "Field mean must be provided."
                assert field_std is not None, "Field std must be provided."
                assert len(field_mean) == field_ch, "Provide mean for each channel."
                assert len(field_std) == field_ch, "Provide std for each channel."
                # build from specified mean and std
                field_mean = torch.tensor(field_mean).reshape(
                    1, field_ch, *([1] * len(field_grid))
                )
                field_std = torch.tensor(field_std).reshape(
                    1, field_ch, *([1] * len(field_grid))
                )
        else:
            # no normalization
            field_mean = torch.zeros(1, field_ch, *([1] * len(field_grid)))
            field_std = torch.ones(1, field_ch, *([1] * len(field_grid)))

        # assert statements
        assert all(field_std > 0), "Field std must be positive."

        assert (
            field_mean.ndim == field_std.ndim == field_train.ndim
        ), "Field mean, std, and train data must have the same number "
        "of dimensions."
        assert (
            field_mean.shape[1] == field_std.shape[1] == field_ch
        ), "Field mean, std, and train data must have the same number of channels."

        # normalize
        field_train_norm = (field_train - field_mean) / field_std
        field_val_norm = (field_val - field_mean) / field_std

        return field_train_norm, field_val_norm, field_mean, field_std

    def _normalize_condition(
        self, condition_train: torch.Tensor, condition_val: torch.Tensor
    ):
        """Normalize the condition"""
        _, condition_ch, *condition_grid = condition_train.shape

        if self.normalize.condition.enabled:
            if self.normalize.condition.auto:
                # compute the mean and std
                condition_mean = condition_train.mean(
                    dim=(0, *range(2, condition_train.ndim)), keepdim=True
                )  # mean per channel
                condition_std = condition_train.std(
                    dim=(0, *range(2, condition_train.ndim)), keepdim=True
                )
            else:
                condition_mean = self.normalize.condition.mean
                condition_std = self.normalize.condition.std
                assert condition_mean is not None, "Condition mean must be provided."
                assert condition_std is not None, "Condition std must be provided."
                assert (
                    len(condition_mean) == condition_ch
                ), "Provide mean for each channel."
                assert (
                    len(condition_std) == condition_ch
                ), "Provide std for each channel."

                # build using specified mean and std
                condition_mean = torch.tensor(condition_mean).reshape(
                    1, condition_ch, *([1] * len(condition_grid))
                )
                condition_std = torch.tensor(condition_std).reshape(
                    1, condition_ch, *([1] * len(condition_grid))
                )
        else:
            # no normalization
            condition_mean = torch.zeros(1, condition_ch, *([1] * len(condition_grid)))
            condition_std = torch.ones(1, condition_ch, *([1] * len(condition_grid)))

        # assert statements
        assert all(condition_std > 0), "Field std must be positive."

        assert (
            condition_mean.ndim == condition_std.ndim == condition_train.ndim
        ), "Field mean, std, and train data must have the same number "
        "of dimensions."
        assert (
            condition_mean.shape[1] == condition_std.shape[1] == condition_ch
        ), "Field mean, std, and train data must have the same number of channels."

        # normalize
        condition_train_norm = (condition_train - condition_mean) / condition_std
        condition_val_norm = (condition_val - condition_mean) / condition_std

        return condition_train_norm, condition_val_norm, condition_mean, condition_std

    def _normalize_data(self, train_data: dict, val_data: dict):
        """Normalize the data"""
        # normalize the fields
        field_train = train_data["field"]
        field_val = val_data["field"]
        field_train_norm, field_val_norm, field_mean, field_std = self._normalize_field(
            field_train=field_train, field_val=field_val
        )

        # normalize the conditions
        condition_train = train_data["condition"]
        condition_val = val_data["condition"]
        (
            condition_train_norm,
            condition_val_norm,
            condition_mean,
            condition_std,
        ) = self._normalize_condition(
            condition_train=condition_train, condition_val=condition_val
        )

        # check for NaNs and Infs
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

        # create the normalized data dicts
        train_data_norm = {
            "field": field_train_norm,
            "field_domain": train_data["field_domain"],  # same for train and val
            "condition": condition_train_norm,
            "condition_domain": train_data[
                "condition_domain"
            ],  # same for train and val
        }

        val_data_norm = {
            "field": field_val_norm,
            "field_domain": val_data["field_domain"],  # same for train and val
            "condition": condition_val_norm,
            "condition_domain": val_data["condition_domain"],  # same for train and val
        }

        # statistics
        statistics = {
            "field": {"mean": field_mean, "std": field_std},
            "condition": {"mean": field_mean, "std": field_std},
        }

        return train_data_norm, val_data_norm, statistics

    def setup(self, stage=None):
        """Setup the data module"""
        if self.reload:
            raise NotImplementedError("Under construction. Please wait...")
        else:
            # process the operator fields
            processed_data_dict = self._process_operator_fields()
            # split the data
            train_data, val_data = self._split_data(processed_data_dict)
            # normalize the data
            train_data_norm, val_data_norm, statistics = self._normalize_data(
                train_data, val_data
            )
            # set the statistics attribute
            self.statistics = statistics
            # create the datasets
            self.train_set = TensorDataset(
                train_data_norm["field"],
                train_data_norm["condition"],
                train_data[
                    "LF_field"
                ],  # to add back the low fidelity field during inference
            )
            self.val_set = TensorDataset(
                val_data_norm["field"],
                val_data_norm["condition"],
                val_data[
                    "LF_field"
                ],  # to add back the low fidelity field during inference
            )
            # create the domain tensors
            self.domains = {
                "field": train_data_norm["field_domain"],  # same for train and val
                "condition": train_data_norm[
                    "condition_domain"
                ],  # same for train and val
            }
            # save
            torch.save(self.train_set, self.file_paths["datasets"]["train"])
            torch.save(self.val_set, self.file_paths["datasets"]["val"])
            torch.save(self.statistics, self.file_paths["statistics"])
            torch.save(self.domains, self.file_paths["domains"])

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
        floral: bool,
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
            floral (bool): If True, use residual learning
        """
        super(GPDataModule, self).__init__()
        self.nx = nx  # Dimensionality of the output features
        self.nc = nc  # Dimensionality of the input features
        self.nd = nd  # Dimensionality of the domain (e.g., 2 for 2D data)
        self.n_samples = n_samples  # Number of samples in the dataset
        self.n_sensors = n_sensors  # Number of sensors in the output field
        self.floral = floral  # If True, use residual learning
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
                "train": "trainset_floral_GP.pt" if self.floral else "trainset_GP.pt",
                "val": "valset_floral_GP.pt" if self.floral else "valset_GP.pt",
            },
            "statistics": "statistics_floral_GP.pt"
            if self.floral
            else "statistics_GP.pt",
            "test_config": "test_config_floral_GP.pt"
            if self.floral
            else "test_config_GP.pt",
            "sensor_locations": "sensor_locations_floral_GP.pt"
            if self.floral
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
        if self.floral:
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
