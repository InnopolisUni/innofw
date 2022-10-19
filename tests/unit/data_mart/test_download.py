# standard libraries

import pytest

# local modules
from innofw.data_mart import download_dataset
from innofw.constants import DefaultS3User, DEFAULT_STORAGE_URL, BucketNames
from tests.utils import is_dir_empty

test_folder_url = f"{DEFAULT_STORAGE_URL}/{BucketNames.data_mart.value}/credit_cards"


@pytest.mark.parametrize(
    ["folder_url", "credentials"], [[test_folder_url, DefaultS3User]]
)
def test_download_dataset(folder_url, tmp_path, credentials):
    assert is_dir_empty(tmp_path), "destination directory should be clean"

    download_dataset(
        folder_url,
        tmp_path,
        credentials.ACCESS_KEY.get_secret_value(),
        credentials.SECRET_KEY.get_secret_value(),
    )

    inner_folders = [file for file in tmp_path.iterdir() if file.suffix != ".zip"]
    assert len(inner_folders) != 1, "folder was not downloaded"

    for folder in inner_folders:
        assert not is_dir_empty(folder), "folder should contain at least one file"
