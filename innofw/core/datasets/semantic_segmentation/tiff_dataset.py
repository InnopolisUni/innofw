#
import logging
from typing import List
from typing import Optional

import numpy as np
import rasterio as rio
from pydantic import FilePath
from pydantic import validate_arguments
from torch.utils.data import Dataset

from innofw.constants import SegDataKeys
from innofw.core.augmentations import Augmentation
from rasterio.plot import reshape_as_raster
#
#


def read_tif(path, channels=None) -> np.ndarray:
    ds = rio.open(path)
    if channels is None:
        ch_num = ds.count
    else:
        ch_num = channels
    return np.dstack([ds.read(i) for i in range(1, ch_num + 1)])


def get_metadata(path):
    meta = rio.open(path).meta
    del meta["nodata"]
    return meta


class SegmentationDataset(Dataset):
    @validate_arguments
    def __init__(
        self,
        images: List[FilePath],
        masks: Optional[List[FilePath]] = None,
        transform=None,
        channels: Optional[
            int
        ] = None,  # todo: add support for Optional[List[int]]
        with_caching: bool = False,
        *args,
        **kwargs,
    ):
        """Dataset reading tif files

        Arguments:
            images - list of image paths
            masks - list of mask path corresponding for each image
            transform - augmentations can be both torchvision or albumentations
            channels - number of channels
            with_caching - allows reading the whole dataset into memory
        """
        self.images = images
        self.masks = masks
        self.channels = channels
        if self.masks is not None:
            if len(self.images) != len(self.masks):
                raise ValueError(
                    "number of images not equal to number of masks"
                )

        self.transform = None if transform is None else Augmentation(transform)

        self.in_mem_images, self.in_mem_masks = None, None
        self.with_caching = with_caching

        if self.with_caching:
            logging.warning(f"Reading all the data into memory")
            try:
                self.in_mem_images = [
                    read_tif(img, self.channels) for img in self.images
                ]
                self.in_mem_masks = [read_tif(mask) for mask in self.masks]
            except MemoryError:  # todo: catch memory error
                pass

    def read_image(self, index):
        if self.with_caching and self.in_mem_images is not None:
            image = self.in_mem_images[index]
        else:
            image = read_tif(self.images[index], self.channels)
        image = np.nan_to_num(image, nan=0)
        return image

    def __getitem__(self, index) -> dict:
        image = self.read_image(index)

        output = dict()

        if self.masks is not None:
            if self.with_caching and self.in_mem_masks is not None:
                mask = self.in_mem_masks[index]
            else:
                mask = read_tif(self.masks[index])

            mask = mask.astype(np.int)  # todo: refactor
            mask = reshape_as_raster(mask)[0]
            if mask.shape[-1] == 1:  # todo: refactor
                mask = np.squeeze(mask, -1)

        if self.transform is not None:  # todo:
            if self.masks is None:
                # image = self.transform(image=image)["image"]
                image = self.transform(image)
            else:
                # out = self.transform(image=image, mask=mask)
                # image, mask = out["image"], out["mask"]
                image, mask = self.transform(image, mask)
        else:
            try:
                image = np.moveaxis(image, 2, 0) 
            except:
                pass
        try:
            image = image.astype(np.float32)  # todo: refactor
        except:
            pass

        output[SegDataKeys.image] = image
        if self.masks is not None:
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, 0)

            output[SegDataKeys.label] = mask

        return output

    def __len__(self):
        return len(self.images)
