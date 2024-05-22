from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import albumentations as albu
import numpy as np
import rasterio as rio
from torch.utils.data import Dataset

from .utils import prep_data


class RasterioDataset(Dataset):
    """
    A class to represent a custom Rasterio Dataset.

    raster_files: Union[List[Path], List[str]]
        directory containing images
    bands_num : int
        number of bands in one image
    mask_files: Optional[Union[List[Path], List[str]]] = None
    transform: albu.Compose
        list of transformations to be applied

    Methods
    -------
    __getitem__(self, idx):
        returns transformed image and mask
    """

    def __init__(self, raster_files: Union[List[Path], List[str]], bands_num: int, mask_files: Optional[Union[List[Path], List[str]]] = None,transform: Union[albu.Compose, None] = None,):
        self.raster_files, self.mask_files, self.bands_num, self.transform = raster_files, mask_files, bands_num, transform


    def __len__(self):
        return len(self.raster_files)

    def __getitem__(self, idx):
        with rio.open(self.raster_files[idx], "r") as raster_file:
            image = np.dstack([raster_file.read(i) for i in range(1, self.bands_num + 1)])

        if self.mask_files is not None:
            with rio.open(self.mask_files[idx], "r") as mask_file:
                mask = mask_file.read(1)
        else:
            mask = None

        output = prep_data(image, mask, self.transform)
        output["names"] = str(self.raster_files[idx])  # .parent.name
        return output
