import wandb
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from .utils_IO import get_logger, printer


def get_checkpointer(path: str):
    """Get a ModelCheckpoint callback for PyTorch Lightning."""
    ckp = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath=path,
        filename="model-{epoch:02d}-{val_loss:.2f}",
    )
    return ckp


def get_trainer(
    config: dict,
    hp_config: wandb.sdk.wandb_config.Config | dict,
    checkpointer: L.pytorch.callbacks.ModelCheckpoint,
    verbose: bool = False,
):
    """Get a PyTorch Lightning Trainer with the specified configuration."""
    # extract config and hp_config
    devices = config.train.get("devices", 1)
    accelerator = config.train.get("accelerator", "cpu")
    logger_name = config.get("logger_name", "default_logger").lower().strip()
    precision = config.train.get("precision", "16-mixed")

    logger_name = logger_name + "_floral" if config.floral else logger_name
    max_epochs = hp_config.get("max_epochs", 100)

    # get logger
    logger = get_logger(logger_name)

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
        precision=precision,
        devices=devices,
        accelerator=accelerator,
        callbacks=[checkpointer, early_stop_callback],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    if verbose:
        printer("==" * 50)
        printer("**" * 10 + "Trainer config" + "**" * 10)
        printer(f"Number of devices: {devices}")
        printer(f"Precision: {precision}")
        printer(f"Running on : {accelerator}")
        printer(f"Logger file: {logger_name}")
        printer(f"Max epochs: {max_epochs}")
        printer("==" * 50)

    return trainer


def make_grid(dims, x_min=0, x_max=1):
    """Creates a 1D or 2D grid based on the list of dimensions in dims.

    Example: dims = [64, 64] returns a grid of shape (64*64, 2)
    Example: dims = [100] returns a grid of shape (100, 1)
    Adapted from: https://github.com/GavinKerrigan/functional_flow_matching
    """
    if len(dims) == 1:
        grid = torch.linspace(x_min, x_max, dims[0])
        grid = grid.unsqueeze(-1)
    elif len(dims) == 2:
        _, _, grid = make_2d_grid(dims)
    return grid


def make_2d_grid(dims, x_min=0, x_max=1):
    """
    Adapted from: https://github.com/GavinKerrigan/functional_flow_matching
    """
    # Makes a 2D grid in the format of (n_grid, 2)
    x1 = torch.linspace(x_min, x_max, dims[0])
    x2 = torch.linspace(x_min, x_max, dims[1])
    x1, x2 = torch.meshgrid(x1, x2, indexing="ij")
    grid = torch.cat(
        (x1.contiguous().view(x1.numel(), 1), x2.contiguous().view(x2.numel(), 1)),
        dim=1,
    )
    return x1, x2, grid
