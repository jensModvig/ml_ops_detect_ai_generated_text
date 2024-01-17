import os
from datetime import datetime
from pathlib import Path

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




def predict(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader) -> None:
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


def evaluate():
    """
    Evaluate the model on the test set.
    """

    ### Model path
    saved_path = "models/2024-01-17/09-31-32/distilbert-base-uncased-epoch=00-val_loss=0.00.ckpt"

    # getting the config path
    relative = os.sep.join(saved_path.split(os.sep)[1:-1])
    config_path = Path("outputs") / relative / ".hydra" / "config.yaml"
    # Load the config (.yaml)
    config = OmegaConf.load(config_path)
    # Print the config
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    # Manage paths
    repo_path, data_path, model_path = get_paths()

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
    predict(model, test_loader)


if __name__ == "__main__":
    """
    Run the prediction with the model and dataloader.     
    """
    evaluate()
    
