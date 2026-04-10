import wandb
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import seed_everything
from .utils_IO import get_logger, printer, print_section, check_keys
from collections.abc import Mapping
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf


def seed_model(seed: int = 42):
    """seed the model"""
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False
    # os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    # os.environ["TORCH_ALLOW_TF32"] = "0"
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def build_checkpointer(config: dict, verbose: bool = False):
    """Get the checkpointer for saving and loading model checkpoints.
    Args:
        config (dict):
        verbose(bool):
            verbose flag
    Returns:
        checkpointer (lightning.pytorch.callbacks.ModelCheckpoint):
            checkpointer
    """
    # get the checkpointer
    ckp_save_path = config.get("checkpoint_save_path", "./experiments/")
    # checkpointer path
    check_keys(config, ["floral"])
    path_identifier = "floral" if config.floral else "flora"
    path = ckp_save_path + path_identifier
    checkpointer = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=path,
        filename="model-{epoch:04d}-{val_loss:.6f}",
    )

    if verbose:
        print_section("Checkpointer config")
        printer(f"Saving checkpoint to path: {path}")
        print_section("Checkpointer config", end=True)

    return checkpointer


@dataclass
class TrainerConfig:
    """Default config for Trainer with override support."""

    # config defaults
    config: dict = field(
        default_factory=lambda: {
            "logger_name": "default_logger",
            "train": {
                "devices": 1,
                "accelerator": "cpu",
                "precision": "32-true",
            },
        }
    )
    # hp_config defaults
    hp_config: dict = field(default_factory=lambda: {"max_epochs": 100})

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


def build_trainer(
    config: dict,
    hp_config: wandb.sdk.wandb_config.Config | dict = None,
    checkpointer=None,
    verbose: bool = False,
    group: str = None,
):
    """Get the trainer for training the model.
    Args:
        config (dict):
            Configuration dictionary containing training parameters.
        hp_config (wandb.sdk.wandb_config.Config | dict):
            Hyperparameter configuration
        checkpointer (lightning.pytorch.callbacks.ModelCheckpoint):
            Checkpointer for saving and loading model checkpoints.
        verbose(bool):
            verbose flag
    Returns:
        trainer (L.Trainer): Trainer object for training the model.
    """
    # merge defaults with user-provided runtime config/hp_config
    config_defaults = TrainerConfig()
    config, hp_config = config_defaults.resolve(config=config, hp_config=hp_config)

    # seed
    seed_model()

    # extract config and hp_config
    devices = config.train.get("devices")
    accelerator = config.train.get("accelerator")
    logger_name = config.get("logger_name").lower().strip()
    precision = config.train.get("precision")
    max_epochs = hp_config.get("max_epochs")

    # get logger
    logger_identifier = "floral" if config.floral else "flora"
    logger_name = logger_name + "_" + logger_identifier
    logger = get_logger(logger_name, group=group)

    # early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        min_delta=1e-4,  # Minimum change to qualify as improvement
        patience=int(
            0.3 * max_epochs
        ),  # Number of epochs with no improvement after which training will stop
        verbose=True,
        mode="min",  # "min" for loss, "max" for accuracy
    )

    # trainer
    trainer = L.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        precision=precision,
        devices=devices,
        accelerator=accelerator,
        callbacks=[checkpointer, early_stop_callback],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        # accumulate_grad_batches=1,
    )

    if verbose:
        print_section("Trainer config")
        printer(f"Number of devices: {devices}")
        printer(f"Precision: {precision}")
        printer(f"Running on : {accelerator}")
        printer(f"Logger file: {logger_name}")
        printer(f"Max epochs: {max_epochs}")
        print_section("Trainer config", end=True)

    return trainer
