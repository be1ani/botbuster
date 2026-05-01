"""Repository-relative paths for data, features, and trained models."""

from pathlib import Path


def repo_root() -> Path:
    """Root of the botbuster repository (parent of the `botbuster` package)."""
    return Path(__file__).resolve().parent.parent


def human_data_dir() -> Path:
    return repo_root() / "data" / "human"


def bot_data_dir() -> Path:
    return repo_root() / "data" / "synthetic"


def default_features_csv() -> Path:
    return repo_root() / "features.csv"


def default_model_path() -> Path:
    return repo_root() / "models" / "v0" / "bot_detection_model.pkl"
