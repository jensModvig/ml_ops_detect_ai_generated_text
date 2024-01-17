import os
from datetime import datetime

from omegaconf import OmegaConf
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

from utilities import get_paths
from data.dataloaders import get_dataloaders
from models.model import TextClassificationModel


def get_callbacks(model_path: str, model_name: str) -> list:
    """
    Get a list of PyTorch Lightning callbacks for model training.

    Parameters:
    - model_path (str): Path to the directory for saving model checkpoints.
    - model_name (str): Name of the model.

    Returns:
    - List: List of PyTorch Lightning callbacks.
    """

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_path,
        filename=model_name + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping_callback = EarlyStopping(
        monitor="train_acc", patience=3, verbose=True, mode="max")

    return [checkpoint_callback, early_stopping_callback]


def get_trainer(config: dict, callbacks: list, experiment_name: str) -> pl.Trainer:
    """
    Get a PyTorch Lightning Trainer object based on the config

    Parameters:
    - config (dict): Configuration dictionary containing parameters.
    - callbacks (list): List of callbacks to use.
    - experiment_name (str): Name of the experiment.

    Returns:
    - trainer (Trainer): PyTorch Lightning Trainer object.
    """
    # Initialize a Lightning Trainer
    logger = pl.loggers.WandbLogger(
        entity=config.wandb.entity,
        project=config.wandb.project,
        name=experiment_name)
    #
    if config.training.use == "steps":
        trainer = pl.Trainer(
            max_steps=config.training.max_steps,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=config.training.log_every_n_steps,
        )
    elif config.training.use == "epochs":
        trainer = pl.Trainer(
            max_epochs=config.training.max_epochs,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=config.training.log_every_n_steps,
        )
    else:
        raise ValueError("Invalid value for config.training.use")
    return trainer



def hydra_path_2_save_path(model_path: str):
    """
    Get the path to the hydra directory and construct model paths.

    Parameters:
    - model_path (str): Path to the model directory.

    Returns:
    - model_path (str): Path to the model directory.
    """
    # Get the path to the hydra directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg['runtime']['output_dir']
    # extract elements from the path and assign them to variables
    experiment_name = "_".join(output_path.split(os.sep)[-2:])
    model_path = model_path / os.sep.join(output_path.split(os.sep)[-2:])
    # create model_path if it does not exist
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return model_path, experiment_name



@hydra.main(config_path="../configs",
            config_name="config.yaml", version_base="1.2")
def train_model(config):
    """
    Run a training, including:
    - Getting the model
    - Getting the data
    - Training the model
    - Saving the model
    """

    #current_datetime = datetime.now()
    #now = current_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    # Chekc if an experiment is present (in struct)
    if "experiment" in config:
        # let parameters in the experiment file overwrite the config file
        config = OmegaConf.merge(config, config.experiment)


    # Print the config
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    # Manage paths
    repo_path, data_path, model_path = get_paths()

    # Get model
    model = TextClassificationModel(
        model_name=config.model.model_name,
        num_labels=config.model.num_labels,
        learning_rate=config.training.learning_rate,
    )

    # Get data
    processed_data_path = data_path / config.training.data_path
    train_loader, val_loader = get_dataloaders(processed_data_path, config)

    # Initialize a Lightning Trainer
    model_path, experiment_name = hydra_path_2_save_path(model_path)
    # Create a unique model name
    callbacks = get_callbacks(model_path, model_name=config.model.model_name)
    trainer = get_trainer(config, callbacks, experiment_name)

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # NOTE: lightning saves the model automatically
    # Save the model
    # save_path = model_path / exp_model_name
    # model.save_pretrained(save_path)



if __name__ == "__main__":
    """
    Run the experiment
    """
    train_model()
