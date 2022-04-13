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
from core.logger.report import report
from core.model.big_resnet import *
from core.model.densenet import DenseNet
from core.model.dsnet import DSNet
from core.model.model_abc import ModelABC
from core.model.resnet import ResNet
from core.runner import Runner
import torchvision.datasets as D

if __name__ == "__main__":

    now_str = datetime.now().strftime("%y%m%d_%H%M%S")  # Same for each run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = Runner(device=device, runname=now_str)

    print("Using GPU" if torch.cuda.is_available() else "Using CPU")

    model_sizes = [3, 5, 7, 9]
    models = [DSNet, DenseNet, ResNet]
    model_names = ["DSNet", "DenseNet", "ResNet"]

    k_fold_n = 5
    # Put the k fold loop outside so that all the model will be run at least once.
    for k in range(5):
        for model_size in model_sizes:
            for model, model_name in zip(models, model_names):
                model_name = f"{model_name}_{model_size}"
                train_conf = TrainingConfig(
                    get_model=model,
                    model_params={"model_n": model_size, "device": device},
                    dataset_builder=D.CIFAR10,
                    k_fold=k_fold_n,
                    kth_fold=k,
                    n_early_stopping=-1,
                    milestones=[50, 75],
                    n_epochs=2,
                    is_cuda=torch.cuda.is_available()
                )
                runner.run(train_conf, expname=model_name)

            # Once done with one size, make report
            df = report(os.path.join("_results", now_str))

            df["model_n"] = df["exp_name"].split("_")
            print(df.sort_values("model_n").head(100))
