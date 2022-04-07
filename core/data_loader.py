import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torch.utils.data import random_split, DataLoader
from core.args import TrainingConfig


def data_loader_builder(
        conf: TrainingConfig
):
    """ Given DataConfig, create Pytorch data image loader """

    transform = build_data_transformer(conf)

    train_set = conf.dataset_builder(
        root="data", train=True, download=True, transform=transform
    )
    train_size = len(train_set)

    train_set, validation_set = random_split(
            train_set,
            [
                int(train_size*(1 - conf.valid_ratio)),
                int(train_size*conf.valid_ratio),
                ]
            )

    train_size = len(train_set)
    validation_size = len(validation_set)

    test_set = conf.dataset_builder(
        root="data", train=False, download=True, transform=transform
    )
    test_size = len(test_set)

    batch_size = conf.batch_size
    train_loader = DataLoader(train_set, batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size, pin_memory=True)

    data_loaders = {
        "train": train_loader,
        "test": test_loader,
        "validation": validation_loader
    }
    dataset_sizes = {
        "train": train_size,
        "test": test_size,
        "validation": validation_size
    }

    return data_loaders, dataset_sizes


def build_data_transformer(conf: TrainingConfig):

    if len(conf.pipelines) > 0:
        print("Using default pipelines of lengths: ", conf.pipelines)
        return Compose(conf.pipelines)

    transform = Compose([
        ToTensor(),
        # Normalize(mean, std),
        RandomCrop(32, padding=4, padding_mode='constant'),
        RandomHorizontalFlip(p=0.5)
    ])

    # TODO: Build data transforming pipelines from string specification
    return transform
