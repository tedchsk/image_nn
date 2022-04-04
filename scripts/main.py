import os
from datetime import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from core.args import DataConfig
from core.data_loader import data_loader_builder
from core.logger.default import LoggerDefault
from core.logger.logger_abc import LoggerABC
from core.model.model_abc import ModelABC
from core.model.resnet import ResNet
from core.runner import TrainingConfig, run


if __name__ == "__main__":

    now_str = datetime.now().strftime("%y%m%d_%H%M%S") # Same for each run
    data_conf = DataConfig()
    data_loaders, dataset_sizes = data_loader_builder(data_conf)

    if torch.cuda.is_available():
        print("Using GPUs")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ResNet(model_n=3, device=device).to(device)


    train_conf = TrainingConfig(100, n_early_stopping=-1, milestones=[80])
    run(data_loaders, dataset_sizes, model, train_conf, runname=now_str, expname="no_early_stop")

    train_conf = TrainingConfig(100, n_early_stopping=5, milestones=[80])
    run(data_loaders, dataset_sizes, model, train_conf, runname=now_str, expname="early_stop_5")
    del model # release model from the GPU (hopefully)

    model2 = ResNet(model_n=5, device=device).to(device)
    run(data_loaders, dataset_sizes, model2, train_conf, runname=now_str, expname="model_5")
