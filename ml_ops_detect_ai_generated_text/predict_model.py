import os
from datetime import datetime
from pathlib import Path
import argparse

from omegaconf import OmegaConf
import hydra
import torch
import pytorch_lightning as pl
import wandb
from hydra.experimental import compose
import numpy as np

from utilities import get_paths
from data.dataloaders import get_test_dataloader
from models.model import TextClassificationModel


def setup_arguments() -> argparse.Namespace:
    """Set up the arguments for the script.

    Returns:
        argparse.Namespace: Namespace with arguments
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Your prediction script')
    parser.add_argument('--saved_path', type=str, help='Path to the saved model', required=False, default="")
    args = parser.parse_args()
    return args


def predict(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            logger) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    """
    # Make predictions on the test set
    accs, aucs = [], []
    with torch.no_grad():
        for batch in dataloader:
            acc, auc = model.prediction_step(batch, batch_idx=None)
            accs.append(acc)
            aucs.append(auc)

    # print
    print(f"Accuracy: {np.mean(accs)}")
    print(f"AUC: {np.mean(aucs)}")
    # logging with wandb
    logger.log_metrics({"accuracy": np.mean(accs), "auc": np.mean(aucs)})

def manage_paths(repo_path: Path, model_path: Path) -> str:
    """Manage the paths for the saved model and the config file.

    Args:
        repo_path: path to the repository
        saved_path: path to the saved model

    Returns:
        str: path to the config file
    """
    # Get the saved path
    saved_path = setup_arguments().saved_path
    # If not specified, we'll use the newest model
    if saved_path == "":
        # get newest model, if not specified
        date = os.listdir(model_path)[0]
        time = os.listdir(model_path / date)[-1] # reversed order
        model_name = os.listdir(model_path / date / time)[0]
        saved_path = str(model_path / date / time / model_name)
        #saved_path = "models/2024-01-17/09-31-32/distilbert-base-uncased-epoch=00-val_loss=0.00.ckpt"
    
    print(f"saved_path: {saved_path}")

    # get config file path too
    repo_path = str(repo_path)
    if repo_path in saved_path:
        # remove the full repo path
        saved_path = saved_path.replace(repo_path, "")[1:]
    # getting the config path
    intermediate = os.sep.join(saved_path.split(os.sep)[1:-1])
    config_path = Path("outputs") / intermediate / ".hydra" / "config.yaml"
    # get config file path too
    repo_path = str(repo_path)
    if repo_path in saved_path:
        # remove the full repo path
        saved_path = saved_path.replace(repo_path, "")[1:]
    # getting the config path
    intermediate = os.sep.join(saved_path.split(os.sep)[1:-1])
    config_path = Path("outputs") / intermediate / ".hydra" / "config.yaml"
    print(f"config_path: {config_path}")
    return saved_path, config_path


def evaluate():
    """
    Evaluate the model on the test set.
    """
    # Create a dummy wandb logger
    wandb_logger = pl.loggers.WandbLogger(
        project="ml_ops_detect_ai_generated_text",
        entity="detect_ai_generated_text",
        offline=True,
    )

    # Manage paths
    repo_path, data_path, model_path = get_paths()

    ### Get the saved path
    saved_path, config_path = manage_paths(repo_path, model_path)
    # Load the config (.yaml)
    config = OmegaConf.load(config_path)
    # Print the config
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    ### Get data
    processed_data_path = data_path / config.training.data_path
    test_loader = get_test_dataloader(processed_data_path, config)

    ### Get model
    # Load the model state dict
    state_dict = torch.load(saved_path)
    # Create an instance of the model
    model = TextClassificationModel(
        model_name=config.model.model_name,
        num_labels=config.model.num_labels,
        learning_rate=config.training.learning_rate,
    )
    # Load the state dict into the model
    model.load_state_dict(state_dict['state_dict'], strict=False)
    # strict=False because the model is saved with the lightning module, only match saved keys (i.e. classification layer)

    ### Run prediction
    model.eval()
    predict(model, test_loader, logger=wandb_logger)



if __name__ == "__main__":
    """
    Run the prediction with the model and dataloader.     
    """
    evaluate()
    
