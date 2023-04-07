import pytest
import os
import tempfile
import rasterio
from pathlib import Path
from unittest.mock import MagicMock

from innofw.utils.data_utils.preprocessing.band_composer import BaseBandComposer, BandComposer


@pytest.fixture
def sample_band_files(tmp_path):
    """Creates sample band files in a temporary directory."""
    channels = ["RED", "GRN", "BLU"]
    for ch in channels:
        file_path = tmp_path / f"band_{ch.lower()}.tif"
        with rasterio.open(file_path, "w") as f:
            f.write([])
    return tmp_path


class TestBaseBandComposer:
    """Tests for the BaseBandComposer class."""

    def test_map_band_idx2str(self):
        composer = BaseBandComposer(band_mapping={0: "RED", 1: "GRN", 2: "BLU"})
        assert composer.map_band_idx2str(0) == "RED"
        assert composer.map_band_idx2str(1) == "GRN"
        assert composer.map_band_idx2str(2) == "BLU"

    def test_map_band_name2idx(self):
        composer = BaseBandComposer(band_mapping={0: "RED", 1: "GRN", 2: "BLU"})
        assert composer.map_band_name2idx("RED") == 0
        assert composer.map_band_name2idx("GRN") == 1
        assert composer.map_band_name2idx("BLU") == 2
