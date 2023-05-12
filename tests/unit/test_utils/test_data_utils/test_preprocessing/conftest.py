import pytest
from pathlib import Path
from tests.utils import get_test_folder_path
from innofw.constants import PathLike
from innofw.utils.data_utils.preprocessing.crop_raster import *
from innofw.utils.data_utils.preprocessing.raster_handler import *
import numpy

@pytest.fixture
def sample_data():
    """Create sample raster files for testing"""
    data_folder = get_test_folder_path() / 'data/images/other/satellite_cropped/prepared/one'
    # Create sample rasters
    for i, filename in enumerate(["BLU.jp2", "GRN.jp2", "NIR.jp2", "RED.jp2"]):
        with rasterio.open(data_folder / filename, "w", driver="JP2OpenJPEG",
                           height=1024, width=1024, count=1, dtype=rasterio.uint8) as dst:
            dst.write(i * 10 * numpy.ones((1024, 1024), dtype=rasterio.uint8), 1)

    return data_folder


@pytest.fixture
def dataset(temp_dir):
    dst_path = temp_dir / "test.tif"
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
