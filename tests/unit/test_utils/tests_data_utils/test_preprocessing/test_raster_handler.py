import tempfile
from pathlib import Path

import pytest
import rasterio as rio
from rasterio.crs import CRS

from innofw.utils.data_utils.preprocessing.raster_handler import *


class TestRasterDataset:
    @pytest.fixture
    def temp_file(self):
        with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
            yield Path(tmp.name)

    @pytest.fixture
    def metadata(self):
        return {
            "driver": "GTiff",
            "height": 10,
            "width": 10,
            "count": 1,
            "dtype": "uint8",
            "crs": CRS.from_epsg(4326),
            "transform": rio.transform.from_bounds(0, 0, 1, 1, 10, 10),
        }

    def test_create_dataset(self, temp_file, metadata):
        dataset = RasterDataset(temp_file, metadata)
        assert dataset.ds.crs == metadata["crs"]
        dataset.close()

    def test_add_band(self, temp_file, metadata):
        dataset = RasterDataset(temp_file, metadata)

        # create a test band
        with rio.open(temp_file, "w", **metadata) as f:
            f.write(f.read(1) + 1, 2)

        # add the band to the dataset
        dataset.add_band(temp_file, 2)

        # check that the band was added and has the correct data
        assert dataset.ds.count == 2
        assert dataset.ds.read(2).max() == 2

        dataset.close()
