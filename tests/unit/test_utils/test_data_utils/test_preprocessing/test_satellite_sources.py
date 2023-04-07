import tempfile
from typing import List
from pathlib import Path
from xml.etree import ElementTree
from pydantic import DirectoryPath, FilePath
from tests.utils import get_test_folder_path

import pytest

from innofw.utils.data_utils.preprocessing.satellite_sources import *


class TestSentinel2:
    @pytest.fixture
    def sentinel2(self) -> Sentinel2:
        tmpdir = os.path.join(
            get_test_folder_path(), "data/images/other/satellite_cropped"
        )
        folder = Path(tmpdir)
        folder.mkdir()
        # Create a dummy metadata file
        metadata_file = folder / "MTD_MSIL1C.xml"
        root = ElementTree.Element("root")
        child1 = ElementTree.SubElement(root, "IMAGE_FILE")
        child1.text = "test_bands.tif"
        tree = ElementTree.ElementTree(root)
        tree.write(metadata_file)
        # Create a dummy band file
        band_file = folder / "test_bands.tif"
        band_file.touch()
        # Create an instance of Sentinel2
        return Sentinel2(folder)


class TestLandsat8:
    @pytest.fixture
    def landsat8(self) -> Landsat8:
        tmpdir = os.path.join(
            get_test_folder_path(), "data/images/other/landsat8"
        )
        folder = Path(tmpdir)
        folder.mkdir()
        # Create a dummy metadata file
        metadata_file = folder / "test_MTL.txt"
        metadata_file.touch()
        # Create an instance of Landsat8
        return Landsat8(folder)

