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
    dataset_builder: Callable = D.CIFAR10
    pipelines: List = field(default_factory=list)
    test_pipelines: List = field(default_factory=list)
    name: str = "default"
    batch_size: int = 128
    # test_ratio: float = 0.2  # (test + valid = 1.0)
    valid_ratio: float = 0.1  # (train + valid = 1.0)
    small: bool = False

    # Training specific
    n_epochs: int = 180
    optimizer: Callable = optim.SGD  # or "adam"
    # Optimizer
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    # Scheduler
    milestones: List[int] = field(default_factory=list)
    gamma: float = 0.1
    n_early_stopping: int = 5  # Set to -1 is don't want to early stopping
    k_fold: int = 1
    # The k-th fold (leave the hadling of num_kfold to be outside the TrainingConfig)
    # If this is set, k_fold will be ignored (will only run 1 fold)
    kth_fold: int = -1

    def build_optimizer(self, params):
        return self.optimizer(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def build_scheduler(self, optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)

    # Verbose
    n_epochs_per_print: int = 5  # -1 to not print at all, 1 to print at every epoch
