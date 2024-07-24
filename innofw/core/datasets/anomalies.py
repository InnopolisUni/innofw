from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class AnomaliesDataset(Dataset):
    """
    A class to represent a custom ECG Dataset.

    data_path: str
        path to folder with structure:
        data_path/images/
        data_path/labels/ (optional)

    augmentations: transforms to apply on images

    add_labels: whether to return anomaly segmentation with the image

    Methods
    -------
    __getitem__(self, idx):
        returns X-features, and Y-targets (if the dataset is for testing or validation)
    """

    def __init__(self, data_path, augmentations, add_labels=False):
        self.images = list(Path(data_path + '/images').iterdir())
        self.add_labels = add_labels
        self.augmentations = augmentations
        if self.add_labels:
            self.labels = list(Path(data_path + '/labels').iterdir())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imagePath = self.images[idx]
        image = cv2.imread(str(imagePath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.add_labels:
            return image
        mask = cv2.imread(str(self.labels[idx]), 0)
        image, mask = self.augmentations(image, mask)
        mask = mask[None, :]

        return image, mask
