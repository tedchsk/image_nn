import typing
from os import PathLike
from typing import List
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Class for data loader configuration"""
    data_path: typing.Union[str, bytes, PathLike]
    name: str = "default"


@dataclass
class ModelConfig:
    name: str = "default"
    layers: List = field(default_factory=list)


@dataclass
class EvaluationConfig:
    verbose: bool = False
    save_model: bool = False
    save_losses: bool = False
