## Neural Network Code Planning



### Training Components

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
