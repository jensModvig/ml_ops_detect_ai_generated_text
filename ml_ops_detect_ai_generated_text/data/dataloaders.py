import torch


def get_dataloaders(processed_data_path: str, config: dict) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """
    Get dataloaders for training and validation data

    Parameters:
    - processed_data_path (str): Path to the processed data directory.
    - config (dict): Configuration dictionary containing parameters.

    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    """

    # Get datasets
    train_data_path = processed_data_path / "train_dataset.pt"
    train_dataset = torch.load(train_data_path)
    val_data_path = processed_data_path / "val_dataset.pt"
    val_dataset = torch.load(val_data_path)

    # Get dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        persistent_workers=True,
    )

    return train_loader, val_loader

def get_test_dataloader(processed_data_path: str, config: dict) -> torch.utils.data.DataLoader:
    """
    Get dataloaders for test data

    Parameters:
    - processed_data_path (str): Path to the processed data directory.
    - config (dict): Configuration dictionary containing parameters.

    Returns:
    - test_loader (DataLoader): DataLoader for the test dataset.
    """

    # Get datasets
    test_data_path = processed_data_path / "test_dataset.pt"
    test_dataset = torch.load(test_data_path)

    # Get dataloaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        persistent_workers=True,
    )

    return test_loader
