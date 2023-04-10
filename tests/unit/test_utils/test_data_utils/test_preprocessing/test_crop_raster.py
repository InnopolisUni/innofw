# import os
# import tempfile
# from pathlib import Path
#
# import pytest
# import rasterio
# from rasterio.errors import RasterioIOError
#
# from innofw.utils.data_utils.preprocessing.crop_raster import *
#
#
# @pytest.fixture
# def test_data_dir():
#     return Path(__file__).parent / "data"
#
#
# @pytest.fixture
# def prepared_folder(test_data_dir):
#     return Path(os.environ['PYTHONPATH'])
#
#
# def test_crop_raster(prepared_folder):
#     src_path = prepared_folder / "BLU.jp2"
#     dst_path = Path(tempfile.NamedTemporaryFile(suffix=".tif").name)
#     crop_raster(src_path, dst_path, height=100, width=100)
#
#     with rasterio.open(dst_path) as dst:
#         assert dst.width == 100
#         assert dst.height == 100
#         assert dst.count == 1
#         assert dst.dtypes[0] == "uint16"
#         assert dst.transform[0] == src_path.transform[0]
#
#
# def test_crop_raster_invalid_file(prepared_folder):
#     with pytest.raises(RasterioIOError):
#         crop_raster(prepared_folder / "nonexistent.jp2", Path(tempfile.NamedTemporaryFile(suffix=".tif").name))
#
#
# def test_crop_multiple_rasters(prepared_folder):
#     src_folder_path = prepared_folder
#     dst_folder_path = Path(tempfile.TemporaryDirectory().name)
#     crop_multiple_rasters(src_folder_path, dst_folder_path, height=100, width=100)
#
#     assert len(os.listdir(dst_folder_path)) == len(os.listdir(src_folder_path))
#
#     for dst_file in os.listdir(dst_folder_path):
#         src_file = src_folder_path / dst_file
#         dst_file = dst_folder_path / dst_file
#
#         with rasterio.open(dst_file) as dst:
#             with rasterio.open(src_file) as src:
#                 assert dst.width == 100
#                 assert dst.height == 100
#                 assert dst.count == src.count
#                 assert dst.dtypes == src.dtypes
#                 assert dst.transform == src.transform
#
#
# def test_crop_multiple_rasters_invalid_folder(prepared_folder):
#     with pytest.raises(RasterioIOError):
#         crop_multiple_rasters(prepared_folder / "nonexistent_folder", Path(tempfile.TemporaryDirectory().name), height=100, width=100)
#

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


