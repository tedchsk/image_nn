
import os
from core.result_reader import combine_run_experiments

RESULT_DIR = "_results"

dirs = os.listdir(RESULT_DIR)
latest_dir = sorted(dirs)[-1]
df = combine_run_experiments(os.path.join(RESULT_DIR, latest_dir))

for col in ["train", "test", "val"]:
    df[f"{col}_err"] = 1 - df[f"{col}_acc"]

summarized = df.groupby("exp_name").agg(
        {
            "train_er": ["mean", "std"],
            "val_err": ["mean", "std"],
            "test_err": ["mean", "std"],
            "n_epochs": ["count"],
            "train_time": ["mean", "std"],
            }
        )

print(summarized.head(100))
