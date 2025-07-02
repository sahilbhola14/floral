import os
import os.path as osp
import torch
import json

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime


@rank_zero_only
def printer(message):
    """Print message only on rank 0"""
    print(message)


def make_dirs(dirname) -> None:
    """Create directories if they do not exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def check_path(path: str) -> None:
    """Check if a path exists, raise an error if it does not."""
    assert osp.exists(path), f"Path {path} does not exist"


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


def get_logger(name: str):
    """Get a TensorBoard logger with a unique version based on the current time."""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = TensorBoardLogger("logs", name=name, version=f"{current_time}")
    return logger


def get_path(path: list) -> str:
    """Get the path by joining the elements of the list."""
    return os.path.join(*path)


def save_args(args, path: str) -> None:
    """Save the arguments to a JSON file."""
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved args to {path}")


def save_checkpoint(state, path: str) -> None:
    """Save the model state to a file."""
    torch.save(state, path)
