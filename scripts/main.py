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
    now_str = datetime.now().strftime("%y%m%d_%H%M%S")
    logger = LoggerDefault(os.path.join("_results", now_str))

    data_conf = DataConfig()
    data_loaders, dataset_sizes = data_loader_builder(data_conf)

    if torch.cuda.is_available():
        print("Using GPUs")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ResNet(model_n=3, device=device).to(device)


    train_conf = TrainingConfig(100, n_early_stopping=5)
    run(data_loaders, dataset_sizes, model, logger, train_conf)
