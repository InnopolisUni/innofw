#
from typing import List
from typing import Optional

import numpy as np
import rasterio as rio
from pydantic import FilePath
from pydantic import validate_arguments
from innofw.constants import SegDataKeys
from torch.utils.data import Dataset

#
#


def read_tif(path, channels=None) -> np.array:
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
        *args,
        **kwargs,
    ):
        self.images = images
        self.masks = masks
        self.channels = channels
        assert len(self.images) == len(self.masks)
        self.transform = transform

    def __getitem__(self, index) -> dict:
        image = read_tif(self.images[index], self.channels)
        image = np.nan_to_num(image, nan=0)

        output = dict()

        if self.masks is not None:
            mask = read_tif(self.masks[index])

            mask = mask.astype(np.int)  # todo: refactor
            if mask.shape[-1] == 1:  # todo: refactor
                mask = np.squeeze(mask, -1)

        if self.transform is not None:  # todo:
            if self.masks is None:
                image = self.transform(image=image)
            else:
                out = self.transform(image=image, mask=mask)
                image, mask = out["image"], out["mask"]

        image = np.moveaxis(image, 2, 0)  # todo: refactor
        image = image.astype(np.float32)  # todo: refactor

        output[SegDataKeys.image] = image
        if self.masks is not None:
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, 0)

            output[SegDataKeys.label] = mask

        return output

    def __len__(self):
        return len(self.images)
