# examples/advection/run_floral.py
"""
Flow-matching operator for residual-augmented learning for Darcy flow.
Notes:
    1. To improve reproducibility across architectures, tf32 is not used.
    Emperically, operators were producing higher losses.
"""
import torch
import wandb
import argparse
import yaml
from omegaconf import OmegaConf
from floral.utils import (
    printer,
    print_section,
    check_path,
    build_data_module,
    build_checkpointer,
    build_trainer,
)

from floral.flow import perform_inference, Flow

parser = argparse.ArgumentParser(description="Run advection with specified parameters.")
parser.add_argument(
    "--config",
    type=str,
    default="config_floral.yml",
    help="Path to the configuration file.",
)
parser.add_argument(
    "--hp_config",
    type=str,
    default="config_hyperparameters.yml",
    help="Path to the configuration file.",
)


args = parser.parse_args()
config = OmegaConf.load(args.config)


def print_header(config: dict):
    """Print the Header
    Args:
        config (dict): Configuration dictionary parameters.
    """
    print_section("advection config")
    printer(f"Job name: {config.job_name}")
    printer(f"Configuration file: {args.config}")
    printer(f"Hyperparameter Configuration file: {args.hp_config}")
    printer(f"Tune hyperparameters: {config.tune_hyperparameters}")
    printer(f"Multi-fidelity Flow: {config.floral}")
    printer(f"Number of samples: {config.data.n_samples}")
    print_section("advection config", end=True)


def train_model(hp_config: dict = None):
    """Train the model using the specified configuration and hyperparameters.
    Returns:
        best_model_path (str): Path to the best model checkpoint after training.
        data_module( L.LightningDataModule|None): Data module
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
    data_module = build_data_module(config=config, hp_config=hp_config, verbose=True)
    # checkpointer
    checkpointer = build_checkpointer(config=config)
    # get trainer
    trainer = build_trainer(
        config=config, hp_config=hp_config, checkpointer=checkpointer, verbose=False
    )
    # model
    flow = Flow(
        config=config,
        hp_config=hp_config,
        domain_dict=data_module.domain_dict,
        shape_dict=data_module.shape_dict,
    )
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
    print_section("Training")
    trainer.fit(flow, data_module)

    # return
    if config.tune_hyperparameters:
        # clean up wandb run
        wandb.finish()
    else:
        # best model path
        best_model_path = checkpointer.best_model_path
        printer(f"Best model saved at {best_model_path}")
        return best_model_path, data_module


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
        with open(args.hp_config, "r") as file:
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
        perform_inference(
            best_model_path=best_model_path, data_module=data_module, config=config
        )
