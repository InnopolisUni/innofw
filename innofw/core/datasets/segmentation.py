from pathlib import Path

import cv2
from torch.utils.data import Dataset

from innofw.constants import SegDataKeys


class SegmentationDataset(Dataset):
    """
        A class to represent a custom Segmentation Dataset.

        image_paths: Union[List[Path], List[str]]
            directory containing images
        bands_num : int
            number of bands in one image
        mask_paths: Optional[Union[List[Path], List[str]]] = None
            directory containing masks
        transforms: albu.Compose
            list of transformations to be applied

        Methods
        -------
        __getitem__(self, idx):
            returns transformed image and mask as dict with "scenes" and "labels" keys
    """

    def __init__(self, image_paths, mask_paths, transforms=None):
        self.imagePaths = list(Path(image_paths).iterdir())
        self.maskPaths = list(Path(mask_paths).iterdir())
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        image = cv2.imread(str(imagePath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.maskPaths is None:
            return image
        mask = cv2.imread(str(self.maskPaths[idx]), 0)
        image, mask = self.transforms(image, mask)
        mask = mask[None, :]
        return {SegDataKeys.image: image.float(), SegDataKeys.label: mask.float()}
