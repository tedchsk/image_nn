import os
import numpy as np
import pandas as pd


def combine_run_experiments(run_dir: str) -> pd.DataFrame:

    # Get all experiment names
    exp_names = os.listdir(run_dir)

    # For each run_dir + exp
    results = []
    for exp_name in exp_names:
        exp_dir = os.path.join(run_dir, exp_name)
        result = combine_run_experiment(exp_dir)
        result["exp_name"] = exp_name
        results.append(result)

    return pd.concat(results)


def combine_run_experiment(exp_dir: str) -> pd.DataFrame:
    fold_dirs = os.listdir(exp_dir)

    dfs = []
    for fold in fold_dirs:
        fold_dir = os.path.join(exp_dir, fold)
        # Read three files - logs, summarized, and training_conf (not here for now)
        # For now let's read only logs
        logs = np.load(
            os.path.join(fold_dir, "logs.npy"), allow_pickle=True
        ).flat[0]

        df = pd.DataFrame.from_dict(logs)
        df["fold"] = fold
        dfs.append(df)

    return pd.concat(dfs)
