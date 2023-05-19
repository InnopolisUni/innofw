from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)
import cv2
import torch

from typing import List, Optional
from pathlib import Path
from skimage.draw import polygon2mask
import numpy as np
from torch.utils.data import Dataset
from innofw.constants import SegDataKeys

from torch.utils.data import Dataset
from innofw.constants import SegDataKeys
from skimage.draw import polygon2mask


class StrokeSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None) -> None:
        self.imagePaths = list(Path(image_paths).iterdir())
        self.maskPaths = list(Path(mask_paths).iterdir())
        self.transforms = transforms


    def __len__(self):
        return len(self.imagePaths)


    def __getitem__(self, index):
        imagePath = self.imagePaths[index]
        image = cv2.imread(str(imagePath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.maskPaths is None:
            return image
        
        mask_path = str(imagePath).replace('.jpg', '.txt')
        mask_path = mask_path.replace('image', 'label')
        coordinates = []
        if str(mask_path)[-4:] != '.txt':
            raise NameError('Eror: mask should be .txt format')
        if Path(mask_path).is_file():
            with open(mask_path) as f:
                for line in f:
                    coordinates.append([float(x) for x in line.split()])
        else:
            print(imagePath)

        mask = np.zeros((512, 512), dtype=bool)               
        for contour_coordinates in coordinates:
            contour_coordinates = contour_coordinates[1:]
            sublist = [contour_coordinates[n: n+2] for n in range(0, len(contour_coordinates), 2)]
            int_sublist = [[int(i * 512), int(j * 512)] for (j, i) in sublist]
            mask = np.maximum(polygon2mask((512, 512), np.array(int_sublist)), mask)
        mask = mask.astype(np.float32) 
        if self.transforms != None:
            image, mask = self.transforms(image, mask)
        else:
            image = torch.from_numpy(image)
            image = image.unsqueeze(0).float()
            mask = torch.from_numpy(mask)
        image = torch.div(image, 255)
        
        
        
        mask = mask[None, :]

        return {
            SegDataKeys.image: image.float(),
            SegDataKeys.label: mask.float(),
        }