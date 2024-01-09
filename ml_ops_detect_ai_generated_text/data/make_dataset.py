import subprocess

import numpy as np
import hydra
import pandas as pd
import re
import torch
from transformers import DistilBertTokenizer
from torch.utils.data import TensorDataset

from ml_ops_detect_ai_generated_text.utilities import get_paths


def clean_text(text: str) -> str:
    # RE to remove unnecessary spaces, keep single
    cleaned_text = re.sub(" +", " ", text)
    # lower case
    cleaned_text = cleaned_text.lower()
    return cleaned_text


def process_data(raw_data_path, processed_data_path):
    # Read data
    data = pd.read_csv(raw_data_path / "train_essays.csv")

    # Clean text
    data["text"] = data["text"].apply(clean_text)

    # Load DistilBERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    # Tokenize the entire dataset
    max_length = 512
    tokenized_texts = tokenizer(
        data["text"].tolist(), truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    labels = data["generated"].tolist()

    # Split data
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1
    assert train_size + val_size + test_size == 1

    # Split the tokenized dataset into train, validation, and test sets
    train_size = int(train_size * len(tokenized_texts["input_ids"]))
    val_size = int(val_size * len(tokenized_texts["input_ids"]))
    test_size = len(tokenized_texts["input_ids"]) - train_size - val_size

    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(tokenized_texts["input_ids"]
                     [:train_size]).clone().detach(),
        torch.tensor(tokenized_texts["attention_mask"]
                     [:train_size]).clone().detach(),
        torch.tensor(labels[:train_size]).clone().detach(),
    )

    val_dataset = TensorDataset(
        torch.tensor(
            tokenized_texts["input_ids"][train_size: train_size + val_size]).clone().detach(),
        torch.tensor(tokenized_texts["attention_mask"]
                     [train_size: train_size + val_size]).clone().detach(),
        torch.tensor(labels[train_size: train_size +
                     val_size]).clone().detach(),
    )

    test_dataset = TensorDataset(
        torch.tensor(tokenized_texts["input_ids"]
                     [-test_size:]).clone().detach(),
        torch.tensor(tokenized_texts["attention_mask"]
                     [-test_size:]).clone().detach(),
        torch.tensor(labels[-test_size:]).clone().detach(),
    )

    # Save datasets
    torch.save(train_dataset, processed_data_path / "train_dataset.pt")
    torch.save(val_dataset, processed_data_path / "val_dataset.pt")
    torch.save(test_dataset, processed_data_path / "test_dataset.pt")

    print("Data processed successfully!")


if __name__ == "__main__":
    """
    Data processing script.
    Process the raw data (../raw) into a cleaned format (../processed).
    Store it as train, val and test data.
    """

    # Manage paths
    repo_path, data_path, model_path = get_paths()

    # Check if the raw data exists
    raw_data_path = data_path / "raw" / "llm-detect-ai-generated-text"
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_data_path}. " f"Please download the data using the 'dvc pull'"
        )

    # Create processed data folder
    processed_data_path = data_path / "processed" / "llm-detect-ai-generated-text"
    if not processed_data_path.exists():
        processed_data_path.mkdir(parents=True, exist_ok=True)

    # Process data
    process_data(raw_data_path, processed_data_path)
