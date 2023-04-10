import os
import tempfile
from pathlib import Path

import pytest
import rasterio
import numpy

from innofw.constants import PathLike
from innofw.utils.data_utils.preprocessing.crop_raster import *


@pytest.fixture
def sample_data(tmp_path: Path) -> None:
    """Create sample raster files for testing"""
    data_folder = Path(os.environ['PYTHONPATH']) / 'data'
    data_folder.mkdir()
    # Create sample rasters
    for i, filename in enumerate(["BLU.jp2", "GRN.jp2", "NIR.jp2", "RED.jp2"]):
        with rasterio.open(data_folder / filename, "w", driver="JP2OpenJPEG",
                           height=1024, width=1024, count=1, dtype=rasterio.uint8) as dst:
            dst.write(i * 10 * numpy.ones((1024, 1024), dtype=rasterio.uint8), 1)

    return data_folder


def test_crop_multiple_rasters(sample_data: PathLike) -> None:
    """Test crop_multiple_rasters function"""
    # Define output folder
    out_folder = tempfile.mkdtemp()

    # Call function
    crop_multiple_rasters(sample_data, out_folder, height=512, width=512)

    # Check output folder exists and contains expected number of files
    assert os.path.isdir(out_folder)
    assert len(os.listdir(out_folder)) == 0

    # Check output rasters have correct size and metadata
    for file in os.listdir(out_folder):
        with rasterio.open(os.path.join(out_folder, file)) as src:
            assert src.width == 512
            assert src.height == 512
            assert src.transform[0] == src.transform[4] == 5.0


