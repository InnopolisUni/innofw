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
        if str(data_path).endswith('images') or str(data_path).endswith('labels'):
            data_path = data_path.parent
        self.images = list(Path(str(data_path) + '/images').iterdir())
        self.add_labels = add_labels
        self.augmentations = augmentations
        if self.add_labels:
            self.labels = list(Path(str(data_path) + '/labels').iterdir())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float()
        image = torch.div(image, 255)
        if not self.add_labels:
            return self.augmentations(image) if self.augmentations is not None else image
        mask = cv2.imread(str(self.labels[idx]), 0)
        if self.augmentations is not None:
            image, mask = self.augmentations(image, mask)
        return image, mask
