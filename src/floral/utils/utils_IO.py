import os
import os.path as osp
import torch
import json
import time

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from datetime import datetime
from typing import Callable, Any, Tuple, Dict


@rank_zero_only
def printer(message):
    """Print message only on rank 0"""
    print(message)


def make_dirs(dirname) -> None:
    """Create directories if they do not exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def check_path(path: str, suggestion: str = None) -> None:
    """Check if a path exists, raise an error if it does not."""
    assert path is not None and isinstance(
        path, str
    ), "Path must be a string and not None"
    assert osp.exists(path), f"Path {path} does not exist" + (
        f". Did you mean {suggestion}?" if suggestion else ""
    )


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
    """Get a logger with a unique version based on the current time."""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Use WandbLogger for better integration with Weights & Biases
    logger = WandbLogger(
        name=f"experiment-{current_time}",  # this appears in wandb dashboard
        project=name,
        log_model=False,  # optional: logs model checkpoints as artifacts
    )

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


class Timer:
    def __init__(self):
        self.elapsed = 0.0

    def timeit(self, func: Callable, *args: Tuple, **kwargs: Dict) -> Any:
        """
        Times the execution of a function.

        Args:
            func (Callable): The function to time.
            *args, **kwargs: Arguments to pass to the function.

        Returns:
            The result of the function call.
        """
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        self.elapsed = end - start
        print(f"[Timer] {func.__name__} took {self.elapsed: .6f} seconds")
        return result

    def get_last_time(self) -> float:
        """Returns the last recorded elapsed time."""
        return self.elapsed
