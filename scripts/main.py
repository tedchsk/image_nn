import os
from datetime import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from core.args import TrainingConfig
from core.data_loader import data_loader_builder
from core.logger.default import LoggerDefault
from core.logger.logger_abc import LoggerABC
from core.model.big_resnet import *
from core.model.densenet import DenseNet
from core.model.model_abc import ModelABC
from core.model.resnet import ResNet
from core.runner import Runner
import torchvision.datasets as D

if __name__ == "__main__":

    now_str = datetime.now().strftime("%y%m%d_%H%M%S")  # Same for each run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = Runner(device=device, runname=now_str)

    print("Using GPU" if torch.cuda.is_available() else "Using CPU")

    experiments = [3, 5, 7, 9]
    for model_n in experiments:
        model_name = f"Densenet_{model_n}"
        train_conf = TrainingConfig(
            get_model=DenseNet,
            model_params={"model_n": model_n},
            dataset_builder=D.CIFAR100,
            k_fold=5,
            n_early_stopping=-1,
            milestones=[90, 135]
        )
        runner.run(train_conf, expname=model_name)
