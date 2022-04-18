import torchvision
import itertools
import os
from datetime import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from core.args import TrainingConfig
from core.data_loader import data_loader_builder
from core.logger.logger_abc import LoggerABC
from core.logger.report import report
from core.model.big_resnet import *
from core.model.densenet import DenseNet
from core.model.dsnet import DSNet
from core.model.resnet import ResNet
from core.runner import Runner
import torchvision.datasets as D

if __name__ == "__main__":

    now_str = datetime.now().strftime("%y%m%d_%H%M%S")  # Same for each run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = Runner(device=device, runname=now_str)

    # Data loader
    # dataset_builder, stats = D.CIFAR10, ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    dataset_builder, stats = D.CIFAR100, ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))

    transform = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*stats),
        torchvision.transforms.RandomCrop(32, padding=4, padding_mode='constant'),
        torchvision.transforms.RandomHorizontalFlip(p=0.5)
        ]


    print("Using GPU" if torch.cuda.is_available() else "Using CPU")

    # Grid Search parameters
    models = [(ResNet, "ResNet"), (DSNet, "DsNet"), (DenseNet, "DenseNet")]
    model_sizes = [3, 8, 16]
    batch_sizes = [32, 128]
    learning_rates = [0.1, 0.01]
    hyperparams_combinations = itertools.product(models, model_sizes, batch_sizes, learning_rates)

    k_fold_n = 2
    n_epochs = 2
    # Put the k fold loop outside so that all the model will be run at least once.
    for k in range(k_fold_n):
        for ((model, model_name), model_size, batch_size, lr) in hyperparams_combinations:
            model_name = f"{model.name}_{model_size}"
            train_conf = TrainingConfig(
                get_model=model,
                model_params={"model_n": model_size,
                              "device": device, "num_classes": 100},
                dataset_builder=D.CIFAR100,
                pipelines=transform,
                test_pipelines=transform,
                k_fold=k_fold_n,
                kth_fold=k,
                lr=lr,
                batch_size=batch_size,
                n_early_stopping=-1,
                milestones=[int(0.5 * n_epochs), int(0.75 * n_epochs)],
                n_epochs=n_epochs,
                is_cuda=torch.cuda.is_available(),
                n_epochs_per_print=1
            )
            runner.run(train_conf, expname=model_name)

            # Once done with one size, make report
            df = report(os.path.join("_results", now_str))
            print(df.head(100))
