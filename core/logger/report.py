
import os
import math

from pandas.core.frame import DataFrame
from core.result_reader import combine_run_experiments


def report(run_dir) -> DataFrame:
    df = combine_run_experiments(run_dir)

    for col in ["train", "test", "val"]:
        df[f"{col}_err"] = 1 - df[f"{col}_acc"]

    summarized = df.groupby("exp_name").agg(
        {
            "train_err": ["mean", "std"],
            "val_err": ["mean", "std"],
            "test_err": ["mean", "std"],
            "n_epochs": ["count"],
            "train_time": ["mean", "std"],
        }
    )

    return summarized
