# import pytest
# import numpy as np
# import rasterio as rio
# from rasterio.crs import CRS
# from rasterio.warp import calculate_default_transform
# from rasterio.warp import Resampling
# from tempfile import TemporaryDirectory
# from pathlib import Path
# from innofw.utils.data_utils.preprocessing.raster_handler import *
#
#
# @pytest.fixture
# def test_data():
#     temp_dir = TemporaryDirectory()
#     data_dir = Path(temp_dir.name)
#
#     metadata = {"driver": "GTiff", "count": 1, "dtype": "uint8", "nodata": 0}
#     with rio.open(data_dir / "data.tif", "w", **metadata) as dst:
#         dst.write(np.zeros((10, 10), dtype=np.uint8), 1)
#
#     yield data_dir
#
#     temp_dir.cleanup()
#
#
# def test_RasterDataset_creation(test_data):
#     metadata = RasterDataset.get_file_metadata(test_data / "data.tif")
#     dst_path = str(test_data / "out.tif")
#     ds = RasterDataset(dst_path, metadata=metadata)
#     assert isinstance(ds, RasterDataset)
#     assert Path(dst_path).exists()
#     assert ds.ds.meta == metadata
#     assert ds.ds.count == metadata["count"]
#     assert ds.ds.width == metadata["width"]
#     assert ds.ds.height == metadata["height"]
#     assert ds.ds.dtypes[0] == metadata["dtype"]
#     assert ds.ds.nodata == metadata["nodata"]
#     assert ds.ds.crs == metadata["crs"]
#     ds.close()
#
#
# def test_RasterDataset_add_band(test_data):
#     dst_path = str(test_data / "out.tif")
#     ds = RasterDataset(dst_path)
#     assert ds.ds.count == 1
#
#     metadata = {"count": 1, "dtype": "uint8", "nodata": 0}
#     with rio.open(dst_path, "w", **metadata) as dst:
#         dst.write(np.zeros((10, 10), dtype=np.uint8), 1)
#
#     with pytest.raises(rio.errors.RasterioIOError):
#         ds.add_band(test_data / "data.tif", 2)
#
#     with rio.open(test_data / "data.tif") as src:
#         transform, width, height = calculate_default_transform(
#             src.crs,
#             ds.ds.crs,
#             src.width,
#             src.height,
#             *src.bounds,
#         )
#         reproject(
#             source=rio.band(src, 1),
#             destination=rio.band(ds.ds, 2),
#             src_transform=src.transform,
#             src_crs=src.crs,
#             dst_transform=transform,
#             dst_crs=ds.ds.crs,
#             resampling=Resampling.bilinear,
#         )
#
#     ds.add_band(test_data / "data.tif", 2)
#     assert ds.ds.count == 2
#     assert ds.ds.nodata == metadata["nodata"]
#     assert ds.ds.crs == metadata["crs"]
#     assert ds.ds.width == metadata["width"]
#     assert ds.ds.height == metadata["height"]
#     assert ds.ds.dtypes[0] == metadata["dtype"]
#     assert ds.ds.nodata == metadata["nodata"]
#
#     with ds.ds as src:
#         data = src.read(2)
#         assert np.array_equal(data, np.zeros((10, 10), dtype=np.uint8))
#
#     ds.close()
#
#
# def test_RasterDataset_close(test_data):
#     dst_path = str(test_data / "out.tif")
#     ds = RasterDataset(dst_path)
#     ds.close()
#     assert ds.ds.closed
#     with pytest.raises(ValueError):
#         ds.add_band(test_data / "data.tif", 2)
