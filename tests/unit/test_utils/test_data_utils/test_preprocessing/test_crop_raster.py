import os
import tempfile
from pathlib import Path
import pytest
import rasterio
import numpy
from innofw.constants import PathLike
from innofw.utils.data_utils.preprocessing.crop_raster import *
from tests.utils import get_test_folder_path


def test_crop_multiple_rasters(sample_data: PathLike) -> None:
    """Test crop_multiple_rasters function"""
    # Define output folder
    out_folder = tempfile.mkdtemp()

    # Call function
    crop_multiple_rasters(sample_data, out_folder, height=512, width=512, extension='.jp2')

    # Check output folder exists and contains expected number of files
    assert os.path.isdir(out_folder)
    assert len(os.listdir(out_folder)) == 4

    # Check output rasters have correct size and metadata
    for file in os.listdir(out_folder):
        with rasterio.open(os.path.join(out_folder, file)) as src:
            assert src.width == 512
            assert src.height == 512
