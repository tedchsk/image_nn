import os
from datetime import datetime
from dataclasses import dataclass, field
import time
from typing import Callable, List
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


def run(
        model: ModelABC,
        train_conf: TrainingConfig,
        runname: str=datetime.now().strftime("%y%m%d_%H%M%S"), # Same for each run
        expname: str="untitled" # Different for each configuration
):
    data_loaders, dataset_sizes = data_loader_builder(train_conf)
    logger = LoggerDefault(os.path.join("_results", runname, expname))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = train_conf.build_optimizer(model.parameters())
    scheduler = train_conf.build_scheduler(optimizer)

    min_val_loss = np.Inf
    epochs_no_improve = 0

    for epoch in range(train_conf.n_epochs):
        start_time = time.time()
        print("Epoch {}/{}".format(epoch, train_conf.n_epochs - 1))
        print("-" * 30)

        epoch_loss = {"train": 0.0, "validation": 0.0}
        epoch_acc = {"train": 0.0, "validation": 0.0}

        running_loss = {"train": 0.0, "validation": 0.0}
        running_corrects = {"train": 0, "validation": 0}

        for phase in ["train", "validation"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)

            for data in data_loaders[phase]:
                inputs, labels = data

                inputs = inputs.to(model.device)
                labels = labels.to(model.device)

                optimizer.zero_grad()  # clear all gradients

                outputs = model(inputs)  # batch_size x num_classes
                _, preds = torch.max(outputs.data, 1)  # values, indices
                loss = loss_fn(outputs, labels)

                if phase == "train":
                    loss.backward()  # compute gradients
                    optimizer.step()  # update weights/biases

                running_loss[phase] += loss.data.item() * inputs.size(0)
                running_corrects[phase] += torch.sum(preds == labels.data).item()

            epoch_loss[phase] = running_loss[phase] / dataset_sizes[phase]
            epoch_acc[phase] = running_corrects[phase] / dataset_sizes[phase]

        # Visualize the loss and accuracy values.
        training_info = {
            'train_time': np.round(time.time()-start_time, 5),
            'train_loss': np.round(epoch_loss["train"], 5),
            'train_acc': np.round(epoch_acc["train"], 5),
            'val_loss': np.round(epoch_loss["validation"], 5),
            'val_acc': np.round(epoch_acc["validation"], 5),
        }
        print(training_info)
        logger.on_epoch_end(training_info)
        scheduler.step()

        # Early stopping logic. Wonder how tensorflow write this portion of the training process.
        val_loss = epoch_loss["validation"]
        if val_loss < min_val_loss:
             epochs_no_improve = 0
             min_val_loss = val_loss
        else:
            epochs_no_improve += 1
        if train_conf.n_early_stopping > 0 and epochs_no_improve >= train_conf.n_early_stopping:
            print(f"Early stopped because validation doesn't improve in {train_conf.n_early_stopping} epochs")
            break

    test_info = evaluate_test_set(
        data_loaders["test"], model, optimizer, loss_fn
    )
    logger.on_training_end(test_info)


def evaluate_test_set(test_loader, model, optimizer, loss_fn):

    # evaluating the model with test set
    with torch.no_grad():
        model.eval()
        running_loss = 0
        running_corrects = 0

        start_time = time.time()
        dataset_size = 0
        for data in test_loader:
            inputs, labels = data

            inputs = inputs.to(model.device)
            labels = labels.to(model.device)

            optimizer.zero_grad()  # clear all gradients

            outputs = model(inputs)  # batch_size x num_classes
            _, preds = torch.max(outputs.data, 1)  # values, indices
            loss = loss_fn(outputs, labels)

            running_loss += loss.data.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

            dataset_size += inputs.shape[0]

        # Visualize the loss and accuracy values.
        test_info = {
            'test_time': np.round(time.time()-start_time, 5),
            'test_loss': np.round(running_loss / dataset_size, 5),
            'test_acc': np.round(running_corrects / dataset_size, 5),
        }
        print(test_info)
        return test_info

