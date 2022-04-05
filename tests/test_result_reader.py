import os
from core.result_reader import combine_run_experiments


def test_combine_logger_default_results():
    df = combine_run_experiments("tests/sample_results/220405_012325")
    assert df.shape[0] > 0
