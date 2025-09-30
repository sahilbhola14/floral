# examples/oneDCorr/run_floral.py
"""
Flow-matching operator for residual-augmented learning for 1D toy problem
"""
import torch
import wandb
import argparse
import lightning as L
import yaml
from omegaconf import OmegaConf
from floral.utils import (
    OpDataModule,
    get_checkpointer,
    get_trainer,
    printer,
    check_path,
    Inference,
)
from floral.flow import Flow

parser = argparse.ArgumentParser(description="Run oneDCorr with specified parameters.")
parser.add_argument(
    "--config",
    type=str,
    default="config_floral.yml",
    help="Path to the configuration file.",
)

args = parser.parse_args()
config = OmegaConf.load(args.config)

torch.set_float32_matmul_precision("medium")  # for tensor cores


def print_header(config: dict):
    """Print the Header
    Args:
        config (dict): Configuration dictionary parameters.
    """
    printer("==" * 50)
    printer("Running oneDCorr")
    printer(f"Job name: {config.job_name}")
    printer(f"Configuration file: {args.config}")
    printer(f"Tune hyperparameters: {config.tune_hyperparameters}")
    printer(f"Multi-fidelity Flow: {config.floral}")
    printer(f"Number of samples: {config.data.n_samples}")
    printer("==" * 50)


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


def build_checkpointer(config: dict):
    """Get the checkpointer for saving and loading model checkpoints."""
    # get the checkpointer
    ckp_save_path = config.checkpoint_save_path
    checkpointer = get_checkpointer(
        ckp_save_path + "/floral" if config.floral else ckp_save_path
    )
    return checkpointer


def build_trainer(
    config: dict,
    hp_config: wandb.sdk.wandb_config.Config | dict = None,
    checkpointer=None,
    verbose: bool = False,
):
    """Get the trainer for training the model.
    Args:
        config (dict): Configuration dictionary containing training parameters.
        hp_config (wandb.sdk.wandb_config.Config | dict): Hyperparameter configuration
        checkpointer: Checkpointer for saving and loading model checkpoints.
    Returns:
        trainer (L.Trainer): Trainer object for training the model.
    """
    trainer = get_trainer(
        config=config,
        hp_config=hp_config,
        checkpointer=checkpointer,
        verbose=verbose,
    )

    return trainer


def train_model(hp_config: dict = None):
    """Train the model using the specified configuration and hyperparameters.
    Returns:
        best_model_path (str): Path to the best model checkpoint after training.
    """
    if config.tune_hyperparameters:
        # initialize wandb with the hyperparameter config
        wandb.init()
        hp_config = wandb.config
    else:
        assert (
            hp_config is not None
        ), "Hyperparameter config must be provided for training."
    # data module
    data_module = build_data_module(config=config, hp_config=hp_config)
    # checkpointer
    checkpointer = build_checkpointer(config=config)
    # get trainer
    trainer = build_trainer(
        config=config, hp_config=hp_config, checkpointer=checkpointer
    )
    # model
    flow = Flow(config=config, hp_config=hp_config)
    # if hasattr(flow, "compile") and torch.cuda.is_available():
    #     printer("Compiling the model...")
    #     flow = torch.compile(flow, mode="default")
    # load checkpoint if specified
    if config.checkpoint_load_path is not None:
        check_path(config.checkpoint_load_path)
        best_model_path = config.checkpoint_load_path
        printer(f"Loading checkpoint from {best_model_path}")
        # load the checkpoint
        flow = Flow.load_from_checkpoint(
            best_model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
    # train
    printer("Starting training...")
    trainer.fit(flow, data_module)

    if config.tune_hyperparameters:
        # clean up wandb run
        wandb.finish()
    else:
        # best model path
        best_model_path = checkpointer.best_model_path
        printer(f"Best model saved at {best_model_path}")
        return best_model_path, data_module


def infer_model(best_model_path: str, data_module: L.LightningDataModule):
    """Infererence task"""
    printer("Inference...")
    # load the best model
    best_model = Flow.load_from_checkpoint(
        best_model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    # set model to eval mode
    best_model.eval()
    # enable inference model optimizations
    if hasattr(best_model, "compile") and torch.cuda.is_available():
        # Use torch.compile for PyTorch 2.0+ (significant speedup)
        printer("Compiling the model...")
        best_model = torch.compile(best_model, mode="max-autotune")

    # create inference object
    infer = Inference(
        model=best_model,
        val_set=data_module.val_set,
        statistics=data_module.statistics,
        job_name=config.job_name,
        floral=config.floral,
        generate_config=config.generate,
    )
    # infer the model
    infer()


if __name__ == "__main__":
    # print header
    print_header(config)
    # if tune hyperparameters is True, load the hyperparameter config
    if config.tune_hyperparameters:
        # load the hyperparameter config from a yaml file
        with open("config_sweep.yml", "r") as file:
            hp_config = yaml.safe_load(file)
        # initialize agent
        sweep_id = wandb.sweep(hp_config)
        wandb.agent(sweep_id, function=train_model, count=100)
    else:
        # load the hyperparameter config
        with open("config_hyperparameters.yml", "r") as file:
            hp_config = yaml.safe_load(file)
        # get the best model path (training or evaluation)
        if config.train.stage == "train":
            best_model_path, data_module = train_model(hp_config)
        elif config.train.stage == "eval":
            raise NotImplementedError
            # printer(
            #     "Skipping training and evaluating directly using"
            #     f"{config.checkpoint_load_path}"
            # )
            # assert (
            #     config.dataloader.reload is True
            # ), "Reload must be True for evaluation"
            # assert (
            #     config.checkpoint_load_path is not None
            # ), "checkpoint must be provided in eval mode"
            # # data module
            # data_module = build_data_module(config=config, hp_config=hp_config)
            # # load the checkpoint
            # check_path(config.checkpoint_load_path)
            # best_model_path = config.checkpoint_load_path

        # infer
        infer_model(best_model_path, data_module)
