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

    now_str = datetime.now().strftime("%y%m%d_%H%M%S") # Same for each run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runner = Runner(device=device, runname=now_str)

    train_conf = TrainingConfig(get_model=ResNet, model_params={"model_n": 3, "device": device}, n_early_stopping=5, milestones=[80])
    runner.run(train_conf, expname="early_stop_5")

    train_conf = TrainingConfig(get_model=ResNet, model_params={"model_n": 3, "device": device}, n_early_stopping=-1, milestones=[80])
    runner.run(train_conf, expname="no_early_stop")

    train_conf = TrainingConfig(get_model=ResNet, model_params={"model_n": 5, "device": device}, n_early_stopping=5, milestones=[80])
    runner.run(train_conf, expname="model_5")
