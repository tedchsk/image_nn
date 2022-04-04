import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from core.args import DataConfig
from core.data_loader import data_loader_builder
from core.model.resnet import ResNet

if __name__ == "__main__":
    data_conf = DataConfig()
    data_loaders, dataset_sizes = data_loader_builder(data_conf)

    if torch.cuda.is_available():
        print("Using GPUs")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ResNet(model_n=3, device=device)

    epochs = 100
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80], gamma=0.1)

    for epoch in range(epochs):
        start_time = time.time()
        print("Epoch {}/{}".format(epoch, epochs - 1))
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

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # clear all gradients

                outputs = model(inputs)  # batch_size x num_classes
                _, preds = torch.max(outputs.data, 1)  # values, indices
                loss = loss_fn(outputs, labels)

                if phase == "train":
                    loss.backward()  # compute gradients
                    optimizer.step()  # update weights/biases

                running_loss[phase] += loss.data.item() * inputs.size(0)
                running_corrects[phase] += torch.sum(
                    preds == labels.data).item()

            epoch_loss[phase] = running_loss[phase] / dataset_sizes[phase]
            epoch_acc[phase] = running_corrects[phase] / dataset_sizes[phase]

        # Visualize the loss and accuracy values.
        print({
            'time': np.round(time.time()-start_time, 5),
            'train_loss': np.round(epoch_loss["train"], 5),
            'train_acc': np.round(epoch_acc["train"], 5),
            'val_loss': np.round(epoch_loss["validation"], 5),
            'val_acc': np.round(epoch_acc["validation"], 5),
        })

        scheduler.step()
