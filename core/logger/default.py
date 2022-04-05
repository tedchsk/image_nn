import numpy as np
import os
from os.path import join
from collections import defaultdict
from typing import Dict
from core.args import TrainingConfig
from core.logger.logger_abc import LoggerABC


def avg(l):
    return sum(l) / len(l)


class LoggerDefault(LoggerABC):

    def __init__(self, save_dir: str, override: bool = False):

        self.override = override
        assert override or not os.path.exists(save_dir), \
            "Directory already exists, please  update the file name or set override=True"

        self.save_dir = save_dir

        self.logs = defaultdict(list)

    def on_epoch_end(self, training_info: Dict):
        for k, v in training_info.items():
            self.logs[k].append(v)

    def on_training_end(self, training_end_info: Dict, training_conf: TrainingConfig):
        # Record the whole training_info

        # Whole logs
        os.makedirs(self.save_dir, exist_ok=True)

        whole_logs_dir = join(self.save_dir, "logs.npy")
        with open(whole_logs_dir, "wb") as f:
            np.save(f, self.logs)

        # Summarize the logs + test_info, nah I should make the user make the summarization themselves.
        # Comment this. The whole test_info is the whole train
        # summarized = {k: avg(v) for k, v in self.logs.items()}
        # summarized |= test_info  # Work only in Python 3.9
        summarized_logs_dir = join(self.save_dir, "summarized.npy")
        with open(summarized_logs_dir, "wb") as f:
            np.save(f, training_end_info)

        # Training config
        training_config_dir = join(self.save_dir, "training_config.npy")
        with open(training_config_dir, "wb") as f:
            np.save(f, training_conf)

        return training_end_info
