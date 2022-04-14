## Neural Network Code Planning

### Running Method

##### Through Jupyter Notebook

I have set a Jupyter Notebook on Google Cloud Engine. You can access it here [[Link](http://35.204.111.135:8888/tree/_notebooks)]

Two important notebooks to checkout
1. Training Control Panel [[Link](http://35.204.111.135:8888/notebooks/_notebooks/Training%20Control%20Panel.ipynb)] - a code taken from main.py for convenience. Specify the argument and run the notebook to train the whole experiment flow. 
2. Training Logs Dashboard [[Link](http://35.204.111.135:8888/notebooks/_notebooks/Training%20Logs%20Dashboard.ipynb)] - Code to view the logs logged by the training flow, containing -
    - summarized test errors for each model 
    - losses & accuracies at each epoch as graph 

Feel free to add your own in **"_notebooks"** folder

**Note**: the underscore "_" is used for git to ignore these folders.


### Code Structures

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
  
### Python Environment Setup

```bash
$ make envload # which is equivalent to the line below.
$ conda env create --name python39 --file environment.yml

$ conda activate python39
$ pip install -e . # so that conda sees the project's code.
```

### Experimentation (high-level)

"A experiment" is defined as a one running flow with fixed data, model, hyperparameters.

A combination of these "experiments" is called "run". Most of the time, we want to run multiple experiment settings (e.g., ResNet vs DenseNet vs DsNet) to compare the result between the three experiments. 

`main.py` specifies a "run" setting, consisting of multiple experiments, then executes them. Then, for each experiment, the code log the result as a folder in this format  `"$PROJECT/_results/{RUN_NAME}/{EXP_NAME}/{KFOLD}"`   

**Main thing you'll be specifying for each experiment**  
    - Data loader (CIFAR10? CIFAR100, [other built-in datasets](https://pytorch.org/vision/stable/datasets.html#built-in-datasets))  
    - Data transform policy (padding, horizontal flip?)  
    - Model (ResNet, DenseNet, DSNet, or write your own)  
    - Other hyperparameters (n_epochs, optimizer, learning rate, gamma)  
    
**What the code will provide**  
    1. Training/Validation metrices (losses, acc, time in seconds) for each epoch  
    2. Training/Validation/Test metrices at the end of training  
    - These will be stored in the folder 
**Result visualization**  
    - Follow this notebook on how to visualize the training result [[Link](http://35.204.111.135:8888/notebooks/_notebooks/Training%20Logs%20Dashboard.ipynb)]



### Installation Notes on GCP
(GCP -> Google Cloud Platform).  

- Nvidia drivers: [[Link](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)]  
- Jupyter on GCP: [[Link](https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52)]  
- Add conda env to Jupyter: [[Link](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084)]  

