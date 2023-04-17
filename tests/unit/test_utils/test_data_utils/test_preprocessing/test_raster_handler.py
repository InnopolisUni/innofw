import pytest
import os
from innofw.utils.data_utils.preprocessing.raster_handler import *
from tests.utils import get_test_folder_path


def test_add_band(dataset):
    band_files = [
        get_test_folder_path() / "data/images/other/satellite_cropped/prepared/one/BLU.jp2",
        get_test_folder_path() / "data/images/other/satellite_cropped/prepared/one/GRN.jp2",
        get_test_folder_path() / "data/images/other/satellite_cropped/prepared/one/NIR.jp2",
        get_test_folder_path() / "data/images/other/satellite_cropped/prepared/one/RED.jp2",
    ]

    assert dataset.ds.count == 4
    assert dataset.ds.shape == (100, 100)


def test_reprojection_metadata(dataset):
    file_path = get_test_folder_path() / "data/images/other/satellite_cropped/prepared/one/BLU.jp2"
    metadata = dataset.get_reprojection_metadata(file_path, target_crs_epsg=3857)
    assert metadata["crs"].to_epsg() == 3857
    assert metadata["width"] == 1024
    assert metadata["height"] == 1024
    assert metadata["transform"][0] == pytest.approx(1)
    assert metadata["transform"][3] == pytest.approx(0.0)
    assert metadata["transform"][1] == pytest.approx(0.0)
    assert metadata["transform"][2] == pytest.approx(0)
    assert metadata["transform"][4] == pytest.approx(-1)
    assert metadata["transform"][5] == pytest.approx(1024)


def test_file_metadata(dataset):
    file_path = get_test_folder_path() / "data/images/other/satellite_cropped/prepared/one/BLU.jp2"
    metadata = dataset.get_file_metadata(file_path)
    assert metadata["count"] == 1
    assert metadata["dtype"] == "uint8"
    assert metadata["nodata"] is None
    assert metadata["transform"][0] == pytest.approx(1.0)
    assert metadata["transform"][3] == pytest.approx(0.0)
    assert metadata["transform"][1] == pytest.approx(0.0)
    assert metadata["transform"][2] == pytest.approx(0.0)
    assert metadata["transform"][4] == pytest.approx(1.0)
    assert metadata["transform"][5] == pytest.approx(0.0)
    assert metadata["width"] == 1024
    assert metadata["height"] == 1024
