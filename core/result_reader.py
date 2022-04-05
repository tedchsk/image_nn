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

        if "DS_Store" in fold_dir:
            continue  # Macos stuff

        # Read three files - logs, summarized, and training_conf (not here for now)
        # For now let's read only logs

        summarized_dict = np.load(
            os.path.join(fold_dir, "summarized.npy"), allow_pickle=True
        ).flat[0]

        training_config_dict = np.load(
            os.path.join(fold_dir, "training_config.npy"), allow_pickle=True
        ).flat[0].__dict__

        df = pd.DataFrame.from_dict(
            summarized_dict | pandas_safe_dict(training_config_dict)
        )

        df["fold"] = fold
        dfs.append(df)

    return pd.concat(dfs)


def pandas_safe_dict(training_config):
    new_training_config = {}
    for k, v in training_config.items():
        if type(v) in [list, dict]:
            new_training_config[k] = [str(v)]
        elif k == "get_model":
            continue
        else:
            new_training_config[k] = [v]

    return new_training_config
