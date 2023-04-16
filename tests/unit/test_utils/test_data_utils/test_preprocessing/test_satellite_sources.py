import tempfile
from typing import List
from pathlib import Path
from xml.etree import ElementTree
from pydantic import DirectoryPath, FilePath
from tests.utils import get_test_folder_path

import pytest

from innofw.utils.data_utils.preprocessing.satellite_sources import *


@pytest.fixture
def tempdir():
    with tempfile.TemporaryDirectory() as dir:
        yield Path(dir)

def test_sentinel2(tempdir):
    sentinel2 = Sentinel2(tempdir)
    # Ensure find_metadata_file raises exception if MTD_MSIL1C.xml is not present
    with pytest.raises(ValueError):
        sentinel2.find_metadata_file()
    
    # Write a dummy MTD_MSIL1C.xml file and ensure it is found
    metadata_file = tempdir / "MTD_MSIL1C.xml"
    metadata_file.touch()
    assert sentinel2.find_metadata_file() == metadata_file

def test_landsat8(tempdir):
    landsat8 = Landsat8(tempdir)
    # Ensure find_metadata_file raises exception if MTL.TXT is not present
    with pytest.raises(ValueError):
        landsat8.find_metadata_file()
    
    # Write a dummy MTL.txt file and ensure it is found
    metadata_file = tempdir / "MTL.txt"
    metadata_file.touch()
    assert landsat8.find_metadata_file() == metadata_file
