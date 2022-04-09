
import os
from core.result_reader import combine_run_experiments

RESULT_DIR = "_results"

dirs = os.listdir(RESULT_DIR)
latest_dir = sorted(dirs)[-1]
df = combine_run_experiments(os.path.join(RESULT_DIR, latest_dir))

print(df.shape)
print(df.head(100))

summarized = df.groupby("exp_name").agg(
        {
            "train_acc": ["mean", "std"],
            "val_acc": ["mean", "std"],
            "test_acc": ["mean", "std"],
            "n_epochs": ["max", "count"]
            }
        )

print(summarized.head(100))
