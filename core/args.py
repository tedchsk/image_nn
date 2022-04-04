import typing
from typing import Any, Callable, Dict
from core.model.model_abc import ModelABC
import torchvision
from os import PathLike
from torch import optim
import torchvision.datasets as D
from typing import List, Callable
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Class for training configuration"""

    # Model specific
    get_model: Callable[..., ModelABC] = None
    model_params: Dict[str, Any] = None

    # Data loader specific
    dataset_builder = D.CIFAR10
    pipelines: List = field(default_factory=list)
    name: str = "default"
    batch_size: int = 512
    test_ratio: float = 0.2  # (test + valid = 1.0)
    small: bool = False

    # Training specific
    n_epochs: int = 100
    optimizer: Callable = optim.SGD # or "adam" 
    lr: float = 0.01
    momentum: float = 0.9
    milestones: List[int] = field(default_factory=list)
    gamma: float = 0.1
    n_early_stopping: int = 5 # Set to -1 is don't want to early stopping
    k_fold: int = 1
    def build_optimizer(self, params):
        return self.optimizer(params, lr=self.lr, momentum=self.momentum)
    def build_scheduler(self, optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

    # Verbose
    n_epochs_per_print: int = 5 # -1 to not print at all, 1 to print at every epoch

