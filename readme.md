# A Deeper Look into Dense Shortcut Nets

DenseNets have demonstrated superior performance on vision tasks, but have a
much higher memory footprint than ResNets. DSNets claim to have solved this
trade-off. In this paper, we will take a deeper look at this architecture and compare
it side-by-side to the other two in multiple experiments. We find that DSNets are
indeed signficantly better than ResNets, but only if the model size is large enough.
We also test the effect of different hyperparameters and provide a guideline on
when to use each model based on memory and time requirements.

![model structures](images/model_structures.png)

You can read our full report [here](report.pdf)

---

### Installation
- Make sure you have conda in your local machine, then follow the steps below.
- Make sure you are in the project root directory.

```bash
# create conda envirnoment from environment.yml
$ conda env create --name python39 --file environment.yml
# Activate the created environment
$ conda activate python39
# Set current project path as executable so that conda environment uses the project code.
$ pip install -e .
# All tests should passed if the environment installation is complete.
$ pytest 
```

---

### Usage 

After successfully setting up python environment, use this command to start the training.

```bash
$ python scripts/main.py
```

![train sample](images/code_train_sample.png)

Use this command to see the summary of losses and accuracies of all the models. The training process logs the training once each model finishes the training, so all the finished training models will be shown here.

```bash
$ python scripts/report.py
```

![report sample](images/code_report_sample.png)

---
  

### Code High-Level

```mermaid
classDiagram
  direction RL
  
  class Runner {
    -data
    -model
    -logger
    -run()
  }
  
  class Logger {
    on_epoch_end(training_info: dict)
    on_training_end(training_info: dict)
  }
  
  class TrainingConfig {
    ## Model specific ##
    get_model: Callable[..., ModelABC] = None # E.g., ResNet
    model_params: Dict[str, Any] = None
    ## Data loader specific ##
    dataset_builder: Callable = D.CIFAR10
    pipelines: List
    test_pipelines: List
    name: str = "default"
    batch_size: int = 128
    valid_ratio: float = 0.1  # train + valid = 1.0
    small: bool = False
    ## Training specific ##
    n_epochs: int = 180
    optimizer: Callable = optim.SGD  # or "adam"
    # Optimizer
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0001
    ## Scheduler ##
    milestones: List[int]
    gamma: float = 0.1
    n_early_stopping: int = 5  # Set to -1 is don't want to early stopping
    k_fold: int = 1
    kth_fold: int = -1
    is_cuda: bool = False
  }
  
  class DataLoader{
    -data_config: DataConfig
    __next__()
  }
  
  class TrainingLogs_npy_file{
    Losses and average for each epoch
    Can visualize later using seaborn
  }
  
 class Summarized_npy_file {
   Loss and average on test set 
   computed at the end of training
  }
  
  class TrainingConfigsStr_npy_file {
    The TrainingConfig used for
    this training iteration
  }
  
  class Model{
    model_n: int
    num_classes: int
    forward()
  }
  
  class ResNet
  class DenseNet
  class DsNet
  
  TrainingConfig --> Runner
  
  ResNet --|> Model
  DenseNet --|> Model
  DsNet --|> Model
  Runner --> DataLoader
  Runner --> Model
  Runner --> Logger
  
  Logger --> TrainingLogs_npy_file
  Logger --> Summarized_npy_file
  Logger --> TrainingConfigsStr_npy_file
  
  ```
  
---

### Installation Notes on Google Cloud Platform (GCP)

This project is trained on GCP. Here are links that might help you along the environment setup process.

- Installation Script: [[Link](https://github.com/teerapat-ch/image_nn/blob/master/gce_install_script.sh)]
- Installing Nvidia drivers on GCP: [[Link](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)]  
- Jupyter on GCP: [[Link](https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52)]  
- Adding conda env to Jupyter: [[Link](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084)]  
