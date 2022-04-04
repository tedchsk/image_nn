import typing
import torchvision
from os import PathLike
from typing import List
import torchvision.datasets as D
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Class for data loader configuration"""
    dataset_builder = D.CIFAR10
    pipelines: List = field(default_factory=list)
    name: str = "default"
    batch_size: int = 512
    test_ratio: float = 0.2  # (test + valid = 1.0)
    small: bool = False


@dataclass
class ModelConfig:
    name: str = "default"
    layers: List = field(default_factory=list)


@dataclass
class EvaluationConfig:
    verbose: bool = False
    save_model: bool = False
    save_losses: bool = False
