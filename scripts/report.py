import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
from core.result_reader import combine_run_experiments, combine_run_training_logs

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

# Graph

df2 = combine_run_training_logs(os.path.join(RESULT_DIR, latest_dir))
df2 = df2.reset_index()
df2["fold"] = df2["fold"].apply(int)
df2["train_acc"] = df2["train_acc"].apply(float)
df2["val_acc"] = df2["val_acc"].apply(float)

# A bit hacky
df2["exp_name"] = df2["exp_name"] + "-Train"
sns_pp = sns.lineplot(data=df2[df2["fold"] == 0],
                      x="epoch", y="train_acc", hue="exp_name")
df2["exp_name"] = df2["exp_name"] + "-Valid"
sns_pp = sns.lineplot(data=df2[df2["fold"] == 0],
                      x="epoch", y="val_acc", hue="exp_name")
# plt.show()
plt.savefig("graph.png")
print("Graph created at graph.png")
