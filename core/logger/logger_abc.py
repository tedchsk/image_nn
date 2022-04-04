
from abc import ABC
from typing import Dict


class LoggerABC(ABC):

    def __init__(self):
        pass

    def on_epoch_end(self, training_info: Dict):
        pass

    def on_training_end(self, training_info: Dict):
        pass
