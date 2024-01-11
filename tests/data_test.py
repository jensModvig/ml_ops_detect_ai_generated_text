import torch
import os
import pytest

file_path = "data/processed"
@pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
def test_data():
    trainset = torch.load("data/processed/llm-detect-ai-generated-text/train_dataset.pt")
    valset = torch.load("data/processed/llm-detect-ai-generated-text/val_dataset.pt")
    testset = torch.load("data/processed/llm-detect-ai-generated-text/test_dataset.pt")
    
    assert trainset.tensors[1].shape != testset.tensors[1].shape, "Train and test data have different number of features"
    assert trainset.tensors[1].shape != valset.tensors[1].shape, "Train and validation data have different number of features"
    assert trainset.tensors[1].shape != testset.tensors[1].shape, "Train and test data have same number of samples"
    assert trainset.tensors[0].shape > testset.tensors[0].shape, "Train data has less samples than test data"
    assert trainset.tensors[0].shape > valset.tensors[0].shape, "Train data has less samples than validation data"

    #TODO: Add more tests

if __name__ == "__main__":
    test_data()