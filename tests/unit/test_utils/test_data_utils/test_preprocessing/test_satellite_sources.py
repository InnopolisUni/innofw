import tempfile
from typing import List
from pathlib import Path
from xml.etree import ElementTree
from pydantic import DirectoryPath, FilePath
from tests.utils import get_test_folder_path

import pytest

from innofw.utils.data_utils.preprocessing.satellite_sources import *


# class TestBaseSatelliteSource:
#     def test_find_metadata_file(self):
#         tmpdir = os.path.join(
#             get_test_folder_path(), "data/images/other/satellite_cropped"
#         )
#         src_folder = Path(tmpdir)
#         source = BaseSatelliteSource()
#         with pytest.raises(NotImplementedError):
#             source.find_metadata_file(src_folder)

# def test_parse_metadata_file(self):
#     tmpdir = os.path.join(
#         get_test_folder_path(), "data/images/other/satellite_cropped"
#     )
#     metadata_file = Path(tmpdir) / "metadata.xml"
#     root = ElementTree.Element("root")
#     child1 = ElementTree.SubElement(root, "child1")
#     child1.text = "test"
#     tree = ElementTree.ElementTree(root)
#     tree.write(metadata_file)
#     source = BaseSatelliteSource()
#     with pytest.raises(NotImplementedError):
#         source.parse_metadata_file(metadata_file)


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

    # def test_find_metadata_file(self):
    #     tmpdir = os.path.join(
    #         get_test_folder_path(), "data/images/other/satellite_cropped"
    #     )
    #     sentinel2 = Path(tmpdir)
    #     expected_file = sentinel2.src_folder / "MTD_MSIL1C.xml"
    #     assert sentinel2.find_metadata_file() == expected_file

    # def test_parse_metadata_file(self):
    #     tmpdir = os.path.join(
    #         get_test_folder_path(), "data/images/other/satellite_cropped"
    #     )
    #     sentinel2 = Path(tmpdir)
    #     expected_bands = {"1": sentinel2.src_folder / "test_bands.tif"}
    #     expected_mapping = {"tif": 1}
    #     expected_metadata = {
    #         "bands": expected_bands,
    #         "mapping": expected_mapping,
    #         "num_bands": 1,
    #         "date_acquired": "1970-01-01",
    #     }
    #     metadata_file = sentinel2.find_metadata_file()
    #     metadata = sentinel2.parse_metadata_file(metadata_file)
    #     assert metadata == expected_metadata


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

    def test_find_metadata_file(self, landsat8):
        expected_file = landsat8.src_folder / "test_MTL.txt"
        assert landsat8.find_metadata_file() == expected_file
    #
    # def test_parse_metadata_file(self, landsat8):
    #     metadata = landsat8.parse_metadata_file(landsat8.find_metadata_file())
    #     assert metadata == {
    #         "bands": {},
    #         "mapping": {},
    #         "num_bands": 0,
    #         "date_acquired": "1970-01-01",
    #     }