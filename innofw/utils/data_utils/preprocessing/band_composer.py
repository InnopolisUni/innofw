#
import logging
from abc import ABC
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import rasterio as rio
from fire import Fire
from pydantic import DirectoryPath
from pydantic import FilePath
from pydantic import validate_arguments
from rasterio.crs import CRS

from innofw.utils import get_abs_path
from innofw.utils.data_utils.preprocessing.raster_handler import RasterDataset
from innofw.utils.data_utils.preprocessing.satellite_sources import (
    BaseSatelliteSource,
)
from innofw.utils.data_utils.preprocessing.satellite_sources import Landsat8
from innofw.utils.data_utils.preprocessing.satellite_sources import Sentinel2
from innofw.utils.getters import get_log_dir

#
#
# from innofw.core.types import PathLike


class BaseBandComposer(ABC):
    """
    An abstract class that defines methods of band files composition in a single file

    Attributes
    ----------
    DRIVER: str
        sets up the default rasterio driver

    Methods
    -------
    map_band_idx2str(idx: int) -> str
        returns name of the band given position
    get_band_files(
        src_path: DirectoryPath,
        channels: Union[List[str], Tuple[str]] = ("RED", "GRN", "BLU"),
    ) -> List[FilePath]
        returns all matching band files from a folder
    """

    DRIVER = "GTiff"

    def __init__(
        self,
        band_mapping: Optional = None,
        folder_handler: Optional[BaseSatelliteSource] = None,
        resolution: Tuple[int, int] = (10, 10),
    ):
        self.band_mapping = band_mapping
        if self.band_mapping is None:
            self.reverse_band_mapping = None
        else:
            self.reverse_band_mapping = {
                value: key for key, value in self.band_mapping.items()
            }
        self.folder_handler = folder_handler
        self.resolution = resolution

    def map_band_idx2str(self, idx: int) -> str:
        return self.band_mapping[idx]

    def map_band_name2idx(self, name: str) -> int:
        return self.reverse_band_mapping[name]

    def get_band_files(
        self,
        src_path: DirectoryPath,
        channels: Union[List[str], Tuple[str]] = ("RED", "GRN", "BLU"),
    ) -> List[FilePath]:
        if self.folder_handler is None:
            # find files
            band_files = []
            for ch in channels:
                band = list(src_path.rglob(f"*{ch}*"))
                assert (
                    len(band) == 1
                ), f"number of files with the band name: {ch} should be 1"
                band_files.append(band[0])
            assert len(band_files) == len(
                channels
            ), "number of band files should match to number of bands"
        else:
            # retrieve scene metadata from the folder's metadata file
            source_folder_handler = self.folder_handler(src_path)
            scene_metadata = source_folder_handler.metadata

            # retrieve raster metadata from the first band
            bands: Dict[int, Union[Path, str]] = scene_metadata["bands"]
            # get required channels
            band_files = [
                band_file
                for ch in channels
                for band_index, band_file in bands.items()
                if self.map_band_idx2str(band_index) == ch
            ]

        return band_files

    @validate_arguments
    def compose_bands(
        self,
        src_path: DirectoryPath,
        dst_path: Optional[Union[str, Path]] = None,
        channels: Union[List[str], Tuple[str]] = ("RED", "GRN", "BLU"),
        target_crs_epsg: Optional[int] = None,
    ) -> None:
        """todo: add description"""
        band_files = self.get_band_files(src_path, channels)
        for i in band_files:
            print(f"Source file, {i.name}, shape: {rio.open(i).shape}")
            logging.info(f"Source file, {i.name}, shape: {rio.open(i).shape}")

        first_band_file = band_files[0]
        band_metadata = RasterDataset.get_file_metadata(first_band_file)

        # update count
        band_metadata["count"] = len(band_files)
        # reproject if needed
        if (
            target_crs_epsg is not None
            and CRS.from_epsg(target_crs_epsg) != band_metadata["crs"]
        ):
            band_metadata.update(
                **RasterDataset.get_reprojection_metadata(
                    first_band_file, target_crs_epsg, self.resolution
                )
            )
        # create a target raster
        dataset = RasterDataset(dst_path, band_metadata)
        # write each file into raster at
        for ch_index, ch in enumerate(band_files, start=1):
            dataset.add_band(ch, ch_index)

        dataset.close()


class BandComposer(BaseBandComposer):
    """
        A class that has methods of band files composition in a single file

    Usage:
        src_path = root_path / 'source/first'
        dst_path = root_path / 'tmp/something.tif'

        channels = ['RED', 'GRN', 'BLU']

        band_composer = BandComposer()
        band_composer.compose_bands(src_path, dst_path, channels)

    """

    def __init__(self):
        super().__init__()


class Landsat8BandComposer(BaseBandComposer):
    """
        A class that has methods of landsat8 band files composition in a single file

    Usage:
        l8_src_path = root_path / 'landsat8/cloud_L8_summer_28_perm_Perm2_base'
        l8_dst_path = root_path / 'tmp/l8.tif'

        channels = ['RED', 'GRN', 'BLU']

        landsat8_composer = Landsat8BandComposer()
        landsat8_composer.compose_bands(l8_src_path, l8_dst_path, channels)

    """

    def __init__(self):
        band_mapping = {
            12: None,
            11: None,
            10: None,
            9: None,
            8: None,
            7: None,
            6: None,
            5: "NIR",
            4: "RED",
            3: "GRN",
            2: "BLU",
            1: None,
        }
        super().__init__(band_mapping, Landsat8)


class Sentinel2BandComposer(BaseBandComposer):
    """
        A class that has methods of sentinel2 band files composition in a single file

    Usage:
        s2_src_path = root_path / 'sentinel2/cloud_S2_summer_28_perm_Perm2_base'
        s2_dst_path = root_path / 'tmp/s2.tif'

        channels = ['RED', 'GRN', 'BLU']

        s2_composer = Sentinel2BandComposer()
        s2_composer.compose_bands(s2_src_path, s2_dst_path, channels)

    """

    def __init__(self):
        band_mapping = {
            14: None,
            13: None,
            12: None,
            11: None,
            10: None,
            9: None,
            8: "NIR",
            7: None,
            6: None,
            5: None,
            4: "RED",
            3: "GRN",
            2: "BLU",
            1: None,
        }
        super().__init__(band_mapping, Sentinel2)


@validate_arguments
def compose_bands(
    src_type: str,
    src_path: Path,
    channels: List[str],
    dst_path: Optional[Path] = None,
):
    """Function for"""
    src2cls = {
        "sentinel2": Sentinel2BandComposer,
        "landsat8": Landsat8BandComposer,
    }
    band_composer = src2cls[src_type]()

    src_path = get_abs_path(src_path)

    if dst_path is None:
        dst_path = (
            get_log_dir(
                project="complexing_data",
                stage="infer",
                experiment_name="",
                log_root=get_abs_path("logs"),
            )
            / "complexing_data.tif"
        )

    band_composer.compose_bands(src_path, dst_path, channels)
    logging.info(f"Resulting file shape: {rio.open(dst_path).shape}")
    raster = rio.open(dst_path)

    print(
        f"Resulting file, {dst_path.name}, shape: ({raster.meta['count']}, {', '.join([str(i) for i in raster.shape])})"
    )
    logging.info(
        f"Resulting file, {dst_path.name}, shape: ({raster.meta['count']}, {', '.join([str(i) for i in raster.shape])})"
    )

    print(
        f"Files in the same directory:{[str(f) for f in list(dst_path.parent.iterdir())]}"
    )
    logging.info(
        f"Files in the same directory:{[str(f) for f in list(dst_path.parent.iterdir())]}"
    )


if __name__ == "__main__":
    Fire(compose_bands)
