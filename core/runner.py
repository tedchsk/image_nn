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

SEED_FOR_FOLDS = [42, 124, 6124, 1235, 1265, 1734,
                  134, 16, 12, 61, 123, 643, 6143, 6413, 712, 1024, 612, 6124, 995, 12512, 16141236]


class Runner:

    def __init__(self, device, runname: str) -> None:
        self.device = device
        self.runname = runname

    def run(
            self,
            train_conf: TrainingConfig,
            expname: str = "untitled"  # Different for each configuration
    ):
        print(f"Start training process of exp: {expname}")

        data_loaders, dataset_sizes = data_loader_builder(train_conf)

        for k in range(train_conf.k_fold):
            # Set random to something else
            torch.manual_seed(SEED_FOR_FOLDS[k])

            if train_conf.k_fold == 1:
                # Don't add nested if only 1 fold
                logger = LoggerDefault(os.path.join(
                    "_results", self.runname, expname))
            else:
                print(
                    f"Fold {k + 1} / {train_conf.k_fold}, seed = {SEED_FOR_FOLDS[k]}")
                logger = LoggerDefault(os.path.join(
                    "_results", self.runname, expname, str(k)))

            model = train_conf.get_model(
                **train_conf.model_params).to(self.device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = train_conf.build_optimizer(model.parameters())
            scheduler = train_conf.build_scheduler(optimizer)

            min_val_loss = np.Inf
            epochs_no_improve = 0

            fold_start_time = time.time()
            for epoch in range(train_conf.n_epochs):
                start_time = time.time()
                if epoch % train_conf.n_epochs_per_print == 0:
                    print("Epoch {}/{} - ".format(epoch +
                          1, train_conf.n_epochs), end="")

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
                        _, preds = torch.max(
                            outputs.data, 1)  # values, indices
                        loss = loss_fn(outputs, labels)

                        if phase == "train":
                            loss.backward()  # compute gradients
                            optimizer.step()  # update weights/biases

                        running_loss[phase] += loss.data.item() * \
                            inputs.size(0)
                        running_corrects[phase] += torch.sum(
                            preds == labels.data).item()

                    epoch_loss[phase] = running_loss[phase] / \
                        dataset_sizes[phase]
                    epoch_acc[phase] = running_corrects[phase] / \
                        dataset_sizes[phase]

                scheduler.step()

                # Calculate the loss and accuracy values.
                training_info = {
                    'train_time': np.round(time.time()-start_time, 5),
                    'train_loss': np.round(epoch_loss["train"], 5),
                    'train_acc': np.round(epoch_acc["train"], 5),
                    'val_loss': np.round(epoch_loss["validation"], 5),
                    'val_acc': np.round(epoch_acc["validation"], 5),
                    'epoch': epoch
                }
                if epoch % train_conf.n_epochs_per_print == 0:
                    print(training_info)
                logger.on_epoch_end(training_info)

                # Early stopping logic.
                val_loss = epoch_loss["validation"]
                if val_loss < min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = val_loss
                else:
                    epochs_no_improve += 1
                if train_conf.n_early_stopping > 0 and epochs_no_improve >= train_conf.n_early_stopping:
                    print(
                        f"Early stopped because validation doesn't improve in {train_conf.n_early_stopping} epochs")
                    break

            test_info = self.evaluate_test_set(
                data_loaders["test"], model, optimizer, loss_fn)

            test_info["train_time"] = np.round(
                time.time() - fold_start_time, 5)
            print("Training Done - ", test_info)

            # For now just report the last training_info, better way is to get minimum acc explicitly.
            logger.on_training_end(test_info | training_info, train_conf)

    def evaluate_test_set(self, test_loader, model, optimizer, loss_fn):

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
            return test_info
