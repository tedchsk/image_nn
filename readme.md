## Neural Network Code Planning


### Code Structures

1. Argument parser.
2. Data
3. Model
4. Data -> Model Feeding Part
5. Performance Evaluation (+ records)


```mermaid
classDiagram
  direction RL
  
  class Runner {
    -data
    -model
    -evaluate
  }
  
  class ArgumentParser{
    build_argument()
  }
  
  class DataConfig{
  }
  
  class ModelConfig{
  }
  
  class Experiment{
    -name: str
    -dataLoader: DataLoader
    -model: Model
    -evaluate: Evaluate
  }
  
  class DataLoader{
    -data_config: DataConfig
    __next__()
  }
  
  class Model{
    predict()
  }
  
  class Evaluate{
    on_epoch_end()
    on_training_end()
  }
  
  class ModelBuilder {
    -modelConfig: modelConfig
    get_model()
  }
  
  ArgumentParser --> DataConfig
  ArgumentParser --> ModelConfig
  
  
  
  DataConfig --> DataLoader
  ModelConfig --> ModelBuilder
  ModelBuilder --> Model
  EvaluateConfig --> Evaluate
  
  DataLoader --> Experiment
  Model --> Experiment
  Evaluate --> Experiment
  Experiment --> Runner
  
  ```
  
### Python Environment

```bash
$ make envload # or
$ conda env create --name python39 --file environment.yml
```

Following this: https://goodresearch.dev/setup.html
  
**For myself**
1. eval "$(/Users/time/miniconda/bin/conda shell.zsh hook)"

