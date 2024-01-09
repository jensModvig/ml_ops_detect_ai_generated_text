from omegaconf import OmegaConf
import hydra
import torch
import pytorch_lightning as pl


from ml_ops_detect_ai_generated_text.utilities import get_paths
from ml_ops_detect_ai_generated_text.data.dataloaders import get_dataloaders
from ml_ops_detect_ai_generated_text.models.model import TextClassificationModel



@hydra.main(config_path="../configs", config_name="config.yaml")
def run_experiment(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    # Extract configs
    model_config = config.model
    training_config = config.training

    # Manage paths
    repo_path, data_path, model_path = get_paths()

    # Get model
    model = TextClassificationModel(
        model_name=model_config.model_name,
        num_labels=model_config.num_labels
    )

    # Get data
    processed_data_path = data_path / training_config.data_path
    train_loader, val_loader = get_dataloaders(processed_data_path)

    stop=0


if __name__ == "__main__":
    """
    
    """
    run_experiment()

