import tempfile
from pathlib import Path
import rasterio
from rasterio.windows import Window
import numpy as np
import pytest

from innofw.utils.data_utils.preprocessing.crop_raster import *


class TestCropRaster:
    @pytest.fixture(scope="class")
    def sample_raster(self):
        # Create a temporary raster for testing
        arr = np.random.rand(100, 100)
        profile = rasterio.profiles.DefaultGTiffProfile(count=1, height=100, width=100)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            with rasterio.open(f.name, "w", **profile) as dst:
                dst.write(arr, 1)
            yield f.name
        Path(f.name).unlink()

    def test_crop_raster(self, sample_raster):
        # Crop the sample raster and test the output
        with rasterio.open(sample_raster) as src:
            expected_data = src.read(window=Window(10, 10, 50, 50))
            expected_profile = src.profile.copy()
            expected_profile.update({"height": 50, "width": 50, "transform": src.window_transform(Window(10, 10, 50, 50))})

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            crop_raster(sample_raster, Path(f.name), height=50, width=50)
            with rasterio.open(f.name) as src:
                np.testing.assert_array_equal(src.read(), expected_data)
                assert src.profile == expected_profile

        Path(f.name).unlink()
