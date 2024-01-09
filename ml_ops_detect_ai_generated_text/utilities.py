from pathlib import Path


"""
This script containes utility functions for the project.
"""


def get_paths():
    """
    Get the paths to the repository, data, and model directories.
    """

    repo_path = Path(__file__).resolve().parents[1]
    data_path = repo_path / "data"
    model_path = repo_path / "models"

    return repo_path, data_path, model_path
