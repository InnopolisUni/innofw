import tempfile
from typing import List
from pathlib import Path
from xml.etree import ElementTree
from pydantic import DirectoryPath, FilePath
from tests.utils import get_test_folder_path

import pytest

from innofw.utils.data_utils.preprocessing.satellite_sources import *


def test_sentinel2(temp_dir):
    sentinel2 = Sentinel2(temp_dir)
    # Ensure find_metadata_file raises exception if MTD_MSIL1C.xml is not present
    with pytest.raises(ValueError):
        sentinel2.find_metadata_file()
    
    # Write a dummy MTD_MSIL1C.xml file and ensure it is found
    metadata_file = temp_dir / "MTD_MSIL1C.xml"
    metadata_file.touch()
    assert sentinel2.find_metadata_file() == metadata_file

def test_landsat8(temp_dir):
    landsat8 = Landsat8(temp_dir)
    # Ensure find_metadata_file raises exception if MTL.TXT is not present
    with pytest.raises(ValueError):
        landsat8.find_metadata_file()
    
    # Write a dummy MTL.txt file and ensure it is found
    metadata_file = temp_dir / "MTL.txt"
    metadata_file.touch()
    assert landsat8.find_metadata_file() == metadata_file
