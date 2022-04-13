import os
import math
from core.result_reader import combine_run_experiments

RESULT_DIR = "_results"

dirs = os.listdir(RESULT_DIR)
latest_dir = sorted(dirs)[-1]
df = combine_run_experiments(os.path.join(RESULT_DIR, latest_dir))

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

# summarized["test_err"]["mean"] - 1.67 * summarized["test_err"]["std"] / math.sqrt(summarized["n_epochs"]["count"])

print(summarized.head(100))
