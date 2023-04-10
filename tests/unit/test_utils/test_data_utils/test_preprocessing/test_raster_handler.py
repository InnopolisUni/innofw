import pytest
import os
from innofw.utils.data_utils.preprocessing.raster_handler import *

@pytest.fixture
def dataset(tmp_path):
    dst_path = tmp_path / "test.tif"
    metadata = {
        "driver": "GTiff",
        "count": 4,
        "dtype": "uint16",
        "nodata": 0,
        "transform": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        "width": 100,
        "height": 100,
        "crs": CRS.from_epsg(4326),
    }
    return RasterDataset(dst_path, metadata)


def test_add_band(dataset):
    band_files = [
        os.environ['PYTHONPATH'] + "/BLU.jp2",
        os.environ['PYTHONPATH'] + "/GRN.jp2",
        os.environ['PYTHONPATH'] + "/NIR.jp2",
        os.environ['PYTHONPATH'] +"/RED.jp2",
    ]
    for i, band_file in enumerate(band_files, start=1):
        dataset.add_band(band_file, i)
    assert dataset.ds.count == 4
    assert dataset.ds.shape == (100, 100)


def test_reprojection_metadata(dataset):
    file_path = os.environ['PYTHONPATH'] + "/BLU.jp2"
    metadata = dataset.get_reprojection_metadata(file_path, target_crs_epsg=3857)
    assert metadata["crs"].to_epsg() == 3857
    assert metadata["width"] == 104
    assert metadata["height"] == 105
    assert metadata["transform"][0] == pytest.approx(18.460614)
    assert metadata["transform"][3] == pytest.approx(0.0)
    assert metadata["transform"][1] == pytest.approx(0.0)
    assert metadata["transform"][2] == pytest.approx(6028373.813011)
    assert metadata["transform"][4] == pytest.approx(-18.460614)
    assert metadata["transform"][5] == pytest.approx(7812456.03387041)


def test_file_metadata(dataset):
    file_path = os.environ['PYTHONPATH'] + "/BLU.jp2"
    metadata = dataset.get_file_metadata(file_path)
    assert metadata["count"] == 1
    assert metadata["dtype"] == "uint16"
    assert metadata["nodata"] is None
    assert metadata["transform"][0] == pytest.approx(10.0)
    assert metadata["transform"][3] == pytest.approx(0.0)
    assert metadata["transform"][1] == pytest.approx(0.0)
    assert metadata["transform"][2] == pytest.approx(690280.0)
    assert metadata["transform"][4] == pytest.approx(-10.0)
    assert metadata["transform"][5] == pytest.approx(6350200.0)
    assert metadata["width"] == 100
    assert metadata["height"] == 100
    assert metadata["crs"].to_epsg()
