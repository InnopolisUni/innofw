from tests.utils import get_test_folder_path
import pytest

def test_file_metadata(dataset):
    file_path = get_test_folder_path() / "data/images/other/satellite_cropped/prepared/one/BLU.jp2"
    metadata = dataset.get_file_metadata(file_path)
    assert metadata["width"] == 1024
    assert metadata["height"] == 1024