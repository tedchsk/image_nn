import pytest
from core.args import DataConfig
from core.data_loader import data_loader_builder


@pytest.mark.slow
def test_data_loader_builder():
    data_config = DataConfig()

    loaders, sizes = data_loader_builder(data_config)
    assert sizes["train"] == 50000
    assert sizes["test"] == 2000

    data = loaders["train"].__iter__()
    print(data)
