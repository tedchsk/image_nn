
import os
from core.result_reader import combine_run_experiments

RESULT_DIR = "tests/sample_results"

dirs = os.listdir(RESULT_DIR)
latest_dir = sorted(dirs)[-1]
df = combine_run_experiments(os.path.join(RESULT_DIR, latest_dir))

print(df.shape)

summarized = df.groupby("exp_name").agg({"train_acc": ["mean", "std"], "val_acc": [
    "mean", "std"], "test_acc": ["mean", "std"]})

print(summarized.head(100))
