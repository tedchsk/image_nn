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
from core.model.model_abc import ModelABC
from core.model.resnet import ResNet
from core.runner import Runner
import torchvision.datasets as D

if __name__ == "__main__":

    now_str = datetime.now().strftime("%y%m%d_%H%M%S")  # Same for each run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = Runner(device=device, runname=now_str)

    experiments = [
            "ResNet18", ResNet18,
            "ResNet34", ResNet34,
            "ResNet50", ResNet50,
            "ResNet70", ResNet70,
            "ResNet101", ResNet101
            ]

    for (model_name, model) in experiments:
        train_conf = TrainingConfig(
                dataset_builder=D.CIFAR100
                get_model=ResNet,
                model_params={ "num_classes": 100, "device": device},
                k_fold=5,
                n_early_stopping=-1,
                milestones=[90, 135]
                )
        runner.run(train_conf, expname=model_name)

