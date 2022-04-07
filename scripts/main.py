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
from core.model.model_abc import ModelABC
from core.model.resnet import ResNet
from core.runner import Runner

if __name__ == "__main__":

    now_str = datetime.now().strftime("%y%m%d_%H%M%S")  # Same for each run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = Runner(device=device, runname=now_str)

    for model_n in [3, 5, 7, 9]:
        train_conf = TrainingConfig(
                get_model=ResNet,
                model_params={ "model_n": model_n, "device": device},
                k_fold=5,
                n_early_stopping=10,
                milestones=[80]
                )
        runner.run(train_conf, expname=f"model_n_{model_n}")

