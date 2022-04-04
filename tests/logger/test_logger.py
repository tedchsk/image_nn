import os
from typing import overload
import numpy as np
from core.args import TrainingConfig
from core.logger.default import LoggerDefault


def test_logger_default_on_epoch_end():
    logger = LoggerDefault("", override=True)
    epoch_num = 20

    for _ in range(epoch_num):
        logger.on_epoch_end({"acc": 0.2, "train_loss": 0.125})

    expected = {
        "acc": [0.2] * epoch_num,
        "train_loss": [0.125] * epoch_num
    }

    assert logger.logs == expected


def test_logger_default_on_training_end(tmpdir):
    # Tmp dir
    logger = LoggerDefault(tmpdir, override=True)

    for _ in range(20):
        logger.on_epoch_end({"acc": 0.2, "train_loss": 0.125})

    expected = {
        "acc": 0.2,
        "train_loss": 0.125,
        "test_acc": 0.9
    }

    summarized = logger.on_training_end({"test_acc": 0.9}, TrainingConfig())

    for k in expected:
        np.testing.assert_almost_equal(summarized[k], expected[k])

    # Check if file got cerated
    assert os.path.exists(os.path.join(tmpdir, "logs.npy"))
    assert os.path.exists(os.path.join(tmpdir, "summarized.npy"))
