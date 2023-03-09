# standard libraries
from pathlib import Path

import pytest
from omegaconf import DictConfig

from tests.utils import get_test_data_folder_path
from tests.utils import is_dir_empty

# third-party libraries
# local modules


@pytest.fixture
def test_dataset_folder() -> Path:
    return get_test_data_folder_path() / "tabular/regression/house_prices"


def test_upload_dataset(test_dataset_folder, tmp_path):
    config_save_path = tmp_path / "config/test.yaml"
    remote_save_path = Path("test_dataset/one")

    assert is_dir_empty(tmp_path)
    assert not config_save_path.exists()

    config_args = DictConfig(
        {
            "name": "some dataset",
            "description": "some dataset",
            "_target_": "innofw.core.datamodules.something",
            "task": ["something"],
            "markup_info": "something",
            "date_time": "something",
        }
    )
    # url = upload_dataset(
    #     test_dataset_folder,
    #     config_save_path,
    #     remote_save_path,
    #     DefaultS3User,
    #     **config_args,
    # )
    #
    # assert config_save_path.exists()
    # assert url is not None
