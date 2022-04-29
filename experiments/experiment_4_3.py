import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch import Tensor 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
from typing import Type, Any, Callable, Union, List, Optional
import pickle
from core.model import ResNet, DenseNet, DSNet
from pathlib import Path
from scipy import stats


def read_confidence_bound_results(args):
    with open(args.output, 'rb') as f:
        results = pickle.load(f)
        
    model_names = ['DSNet', 'ResNet', 'DenseNet']

    test_results = {}
    for model_name in model_names:
        test_results[model_name] = []
        for n in range(args.n_experiments):
            test_results[model_name].append(results[model_name][n]['test']['test_acc'])
            
    mean_std = {}
    for model_name, test_accuracies in test_results.items():
        m = np.mean(test_accuracies)
        s = np.std(test_accuracies)
        mean_std[model_name] = (m, s)
        print('{}: mean={:.4f}, std={:.4f}'.format(model_name, m, s))

    # T-tests:
    print('ResNet vs DSNet:')
    print(stats.ttest_ind(test_results['ResNet'], test_results['DSNet'], equal_var=False, alternative='less'))
    print('DSNet vs DenseNet:')
    print(stats.ttest_ind(test_results['DSNet'], test_results['DenseNet'], equal_var=False, alternative='less'))

def main(args):
    if torch.cuda.is_available():
        print("Using GPUs")
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")

    torch.manual_seed(43)
    batch_size = 128

    ### for CIFAR 10
    # stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ## for CIFAR 100
    stats = ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*stats),
        torchvision.transforms.RandomCrop(32, padding=4, padding_mode='constant'),
        torchvision.transforms.RandomHorizontalFlip(p=0.5)
    ])

    train_set = torchvision.datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
    # For testing purposes:
    # train_set, _ = torch.utils.data.random_split(train_set, [1, len(train_set)-1]) # For sanity checking
    train_size = len(train_set)
    test_set = torchvision.datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
    test_set, validation_set = torch.utils.data.random_split(test_set, [5000, 5000])
    test_size = len(test_set)
    validation_size = len(validation_set)


    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, num_workers=4, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size, num_workers=4, pin_memory=True)

    data_loaders = {"train": train_loader, "test": test_loader, "validation": validation_loader}
    dataset_sizes = {"train": train_size, "test": test_size, "validation": validation_size}
    print(dataset_sizes)


    #### Train Configurations, based on DSNet and ResNet paper
    model_n = args.model_size
    epochs = 100
    milestones = [int(epochs*0.5), int(epochs*0.75)]
    momentum = 0.9
    weight_decay = 0.0005
    gamma = 0.1
    lr = 0.1

    Path("./_results").mkdir(exist_ok=True)

    f1 = open('./_results/results_1.txt', 'w')
    f2 = open('./_results/results_2.txt', 'w')

    model_names = ['DSNet', 'ResNet', 'DenseNet']
    model_fns = [
        lambda: DSNet(model_n=model_n, num_classes=100, device=device),
        lambda: ResNet(model_n=model_n, num_classes=100, device=device),
        lambda: DenseNet(model_n=model_n, growth_rate=16, num_init_features=16, bn_size=2, num_classes=100)
    ]

    results = {model_name: [{'train': [], 'test': None} for _ in range(args.n_experiments)] for model_name in model_names}

    def print1(*args):
        for arg in args:
            f1.write(str(arg) + ' ')
        f1.write('\n')
        f1.flush()

    def print2(*args):
        for arg in args:
            f2.write(str(arg) + ' ')
        f2.write('\n')
        f2.flush()

    for model_name, model_fn in zip(model_names, model_fns):
        print2('Experiments for', model_name)
        for n in range(args.n_experiments):
            model = model_fn()
            model.to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

            ### Train loop + validation/ also test at the end
            print1("Configuration: ", "model:", model_name, " model_n:", model_n, " batch size:", batch_size, 
                " optimizer:SGD", " lr:", lr, " epochs:", epochs)

            print1("----------------------------- Train --------------------------------")
            for epoch in range(epochs):
                start_time = time.time()
                print1("Epoch {}/{}".format(epoch+1, epochs))
                print1("-" * 30)

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

                        optimizer.zero_grad() # clear all gradients

                        outputs = model(inputs) # batch_size x num_classes
                        _, preds = torch.max(outputs.data, 1) # values, indices
                        loss = loss_fn(outputs, labels)

                        if phase == "train":
                            loss.backward()  # compute gradients
                            optimizer.step() # update weights/biases

                        running_loss[phase] += loss.data.item() * inputs.size(0)
                        running_corrects[phase] += torch.sum(preds == labels.data).item()

                    epoch_loss[phase] = running_loss[phase] / dataset_sizes[phase]
                    epoch_acc[phase] =  running_corrects[phase] / dataset_sizes[phase]

                # Visualize the loss and accuracy values.
                results_dic = {
                    'time': np.round(time.time()-start_time, 5),
                    'train_loss': np.round(epoch_loss["train"], 5),
                    'train_acc': np.round(epoch_acc["train"], 5),
                    'val_loss': np.round(epoch_loss["validation"], 5),
                    'val_acc': np.round(epoch_acc["validation"], 5),
                }
                print1(results_dic)
                results[model_name][n]['train'].append(results_dic)

                scheduler.step()



            ### evaluating the model with test set
            print1("----------------------------- Test --------------------------------")
            with torch.no_grad():
                model.eval()
                running_loss = 0
                running_corrects = 0

                for data in test_loader:
                    inputs, labels = data 

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad() # clear all gradients

                    outputs = model(inputs) # batch_size x num_classes
                    _, preds = torch.max(outputs.data, 1) # values, indices
                    loss = loss_fn(outputs, labels)

                    running_loss += loss.data.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()

            # Visualize the loss and accuracy values.
            results_dic = {
            'time': np.round(time.time()-start_time, 5),
            'test_loss': np.round(running_loss/ dataset_sizes['test'], 5),
            'test_acc': np.round(running_corrects/ dataset_sizes['test'], 5),
            }
            print1(results_dic)
            results[model_name][n]['test'] = results_dic

            print2('Experiment {}'.format(n))
            print2('test_acc', np.round(running_corrects/ dataset_sizes['test'], 5))
            
            with open(args.output, 'wb') as f:
                pickle.dump(results, f)
                f.flush()



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default="./result.pk")
    parser.add_argument('--n-experiments', type=int, default=5)
    parser.add_argument('--model_size', type=int, default=3)
    args = parser.parse_args()
    main(args)
    read_confidence_bound_results(args)


