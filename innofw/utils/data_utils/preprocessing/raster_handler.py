"""
Author: Kazybek Askarbek
Date: 01.08.22
Description: File includes raster dataset handler. Current implementation uses rasterio but it could be easily replaced
"""

# third party libraries
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio as rio
from pydantic import FilePath
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject


# local modules
# from innofw.core.types import PathLike


class RasterDataset:
    """Raster dataset class, handles dataset creation, dynamic band addition and sync of nodata value across bands

        Attributes
        ----------
        DN_NODATA: int
            defines values to be used as a replacement of null values in the raster
        DRIVER: str
            sets up the default rasterio driver

        Methods
        -------
        get_file_metadata(file_path: FilePath) -> dict
            Parses file with metadata into a dictionary
        add_band(self, band_path: FilePath, band_index: int) -> None
            Adds a new band inplace into raster. Resamples new band if needed
    """

    DN_NODATA = 0
    DRIVER = "GTiff"

    def __init__(self, dst_path: Union[str, Path], metadata=None):
        dst_path = Path(dst_path)
        dst_path.parent.mkdir(exist_ok=True, parents=True)
        # if (
        #     metadata
        #     and "driver" not in metadata
        #     or (
        #         dst_path.suffix in [".tif", ".tiff"]
        #         and metadata["driver"] != self.DRIVER
        #     )
        # ):
        metadata[
            "driver"
        ] = self.DRIVER  # todo: consider cases when other drivers needed

        self.ds = rio.open(dst_path, "w+", **metadata)

    @staticmethod
    def get_file_metadata(file_path: FilePath) -> dict:
        with rio.open(file_path) as f:
            _metadata = f.meta
        return _metadata

    @staticmethod
    def get_reprojection_metadata(
        file_path: FilePath,
        target_crs_epsg: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> dict:
        with rio.open(file_path) as f:
            left, bottom, right, top = f.bounds

            target_crs = (
                f.crs if target_crs_epsg is None else CRS.from_epsg(target_crs_epsg)
            )

            transform, width, height = calculate_default_transform(
                src_crs=f.crs,
                dst_crs=target_crs,
                width=f.width,
                height=f.height,
                left=left,
                bottom=bottom,
                right=right,
                top=top,
                resolution=resolution,
            )
            metadata = {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
            return metadata

    def add_band(self, band_path: FilePath, band_index: int) -> None:
        """Adds a new band inplace into raster. Resamples new band if needed"""
        with rio.open(band_path) as image_band:
            if self.ds.crs == image_band.crs:
                self.ds.write(image_band.read(1), band_index)
            else:  # todo(qb): I'm unsure about the following code snippet, test thoroughly,
                transform, width, height = calculate_default_transform(
                    image_band.crs,
                    self.ds.crs,
                    image_band.width,
                    image_band.height,
                    *image_band.bounds
                )

                reproject(
                    source=rio.band(image_band, 1),
                    destination=rio.band(self.ds, band_index),
                    src_transform=image_band.transform,
                    src_crds=self.ds.crs,
                    dst_transform=transform,
                    dst_crs=self.ds.crs,
                    resampling=Resampling.bilinear,
                )

    def close(self) -> None:
        self.sync_bands_nodata()
        self.ds.close()

    def sync_bands_nodata(self) -> None:
        pass
        # self._build_nodata_mask()

    #     for band_index in self.ds.indexes:
    #         data = self.ds.read(band_index)
    #         data[self.nodata_mask] = self.DN_NODATA
    #
    #         self.ds.write(data, indexes=band_index)
    #
    # def _build_nodata_mask(self) -> None:
    #     self.nodata_mask = np.zeros((self.ds.count, self.ds.height, self.ds.width), dtype="bool")
    #     for i, band_index in enumerate(self.ds.indexes):
    #         data = self.ds.read(band_index)
    #         # self.nodata_mask[i] = data == -1  # np.logical_or(self.nodata_mask,
