from core.result_reader import combine_run_experiments


df = combine_run_experiments("tests/sample_results/220405_012325")
df.head()

df.shape

df.groupby("exp_name").agg({"train_acc": ["mean", "std"], "val_acc": [
    "mean", "std"], "test_acc": ["mean", "std"]})


df.columns
