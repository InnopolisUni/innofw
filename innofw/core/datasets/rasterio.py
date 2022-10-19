#
from typing import Optional, Union, List
from pathlib import Path

#
import albumentations as albu
import rasterio as rio
import numpy as np
from torch.utils.data import Dataset

#
from .utils import prep_data


class RasterioDataset(Dataset):
    def __init__(
        self,
        raster_files: Union[List[Path], List[str]],
        bands_num: int,
        mask_files: Optional[Union[List[Path], List[str]]] = None,
        transform: Union[albu.Compose, None] = None,
    ):
        # assert len(raster_files) == len(
        #     mask_files
        # ), "Number of rasters and masks should be the same"

        self.raster_files = raster_files
        self.mask_files = mask_files
        self.bands_num = bands_num
        self.transform = transform

    def __len__(self):
        return len(self.raster_files)

    def __getitem__(self, idx):
        with rio.open(self.raster_files[idx], "r") as raster_file:
            bands = [raster_file.read(i) for i in range(1, self.bands_num + 1)]

            image = np.dstack(bands)

        if self.mask_files is not None:
            with rio.open(self.mask_files[idx], "r") as mask_file:
                mask = mask_file.read(1)
        else:
            mask = None

        output = prep_data(image, mask, self.transform)
        output["names"] = str(self.raster_files[idx])  # .parent.name
        return output
