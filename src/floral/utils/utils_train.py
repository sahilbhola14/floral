import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from .utils_IO import get_logger, printer, print_section, check_keys


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
    path = ckp_save_path + "floral" if config.floral else ckp_save_path
    checkpointer = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=path,
        filename="model-{epoch:02d}-{val_loss:.2f}",
    )

    if verbose:
        print_section("Checkpointer config")
        printer(f"Saving checkpoint to path: {path}")
        print_section("Checkpointer config", end=True)

    return checkpointer


def build_trainer(
    config: dict,
    hp_config: wandb.sdk.wandb_config.Config | dict = None,
    checkpointer=None,
    verbose: bool = False,
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

    # extract config and hp_config
    devices = config.train.get("devices", 1)
    accelerator = config.train.get("accelerator", "cpu")
    logger_name = config.get("logger_name", "default_logger").lower().strip()
    precision = config.train.get("precision", "32-true")

    logger_name = logger_name + "_floral" if config.floral else logger_name
    max_epochs = hp_config.get("max_epochs", 100)

    # get logger
    logger = get_logger(logger_name)

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
