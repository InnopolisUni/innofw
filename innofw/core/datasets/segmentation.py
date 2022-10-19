import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import cv2


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.imagePaths = list(Path(image_paths).iterdir())
        self.maskPaths = list(Path(mask_paths).iterdir())
        self.transforms = transforms

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.maskPaths is None:
            return image
        mask = cv2.imread(self.maskPaths[idx], 0)
        image, mask = self.transforms(image, mask)
        mask = mask[None, :]
        return {"scenes": image.float(), "labels": mask.float()}
