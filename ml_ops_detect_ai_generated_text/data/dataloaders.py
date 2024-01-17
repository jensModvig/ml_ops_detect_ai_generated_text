import torch


def get_dataloaders(processed_data_path, config):
    """
    Get dataloaders for training and validation data
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

def get_test_dataloader(processed_data_path, config):
    """
    Get dataloaders for training and validation data
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
