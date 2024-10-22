import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from innofw.constants import SegDataKeys


class ImageFolderInferDataset(Dataset):
    """
    A class to represent a custom Image Dataset for inference.

    image_dir : str
        directory containing images
    transforms : Iterable[albumentations.augmentations.transforms]
    gray : Optional[bool]
        if images in the dir are grayscale

    Methods
    -------
    __getitem__(self, idx):
        returns image read by opencv
    """

    def __init__(self, image_dir, transforms=None, gray=False):
        super().__init__()
        self.image_dir = image_dir
        self.transforms = transforms
        if os.path.isdir(image_dir):
            self.image_names = os.listdir(image_dir)
        self.gray = gray

    def __getitem__(self, index: int):
        if Path(self.image_dir).is_dir():
            image_name = self.image_names[index]
            image = cv2.imread(
                str(Path(self.image_dir, image_name)), cv2.IMREAD_COLOR
            )
        else:
            image = cv2.imread(self.image_dir, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


        if self.transforms != None:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image)
            image = image.unsqueeze(0).float()
            image = torch.div(image, 255)
        return {SegDataKeys.image: image.float()}

    def __len__(self) -> int:
        if os.path.isdir(self.image_dir):
            return len(self.image_names)
        return 1
