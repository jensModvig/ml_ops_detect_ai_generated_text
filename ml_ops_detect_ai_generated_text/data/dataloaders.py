import torch



def get_dataloaders(processed_data_path):
    train_data_path = processed_data_path / 'train_dataset.pt'
    train_dataset = torch.load(train_data_path)
    val_data_path = processed_data_path / 'val_dataset.pt'
    val_dataset = torch.load(val_data_path)

    # Get dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False
    )

    return train_loader, val_loader


