import os
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
import hydra
import torch
import pytorch_lightning as pl

from transformers import DistilBertTokenizer

import sys
sys.path.append("ml_ops_detect_ai_generated_text/models")
from model import TextClassificationModel

def app_predict(text: str, model, access_token) -> bool:

    # Tokenize the text
    # Load DistilBERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", use_auth_token=True, token = access_token)
    # Tokenize the entire dataset
    max_length = 512
    tokenized_texts = tokenizer(
        list(text), truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    

def define_model(model_path):
    # Create an instance of the model
    model = TextClassificationModel(model_name = "distilbert-base-uncased", num_labels = 2, learning_rate = 0.0001)
    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()
    return model

if __name__ == "__main__":
    access_token = "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"
    model_path = "outputs/2024-01-17/19-49-18/distilbert-base-uncased-epoch=00-val_loss=0.00.ckpt"  
    # load the config
    model = define_model(model_path)

    app_predict("This is a test", model)
