import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from innofw.utils.data_utils.preprocessing.dicom_handler import dicom_to_img


class CocoDataset(Dataset):
    """
    A class to represent a Coco format Dataset.
    https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html
    ...

    Attributes
    ----------
    dataframe : pandas.DataFrame
        df with train data info
    image_dir : str
        directory containing images
    transforms : Iterable[albumentations.augmentations.transforms]


    Methods
    -------
    __getitem__(self, index: int):
        returns image and target tensors and image_id

    read_image(self, path):
        reads an image using opencv
    """

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        if "image_id" not in dataframe:
            raise ValueError("image_id column should be in DataFrame")
        self.image_ids = dataframe["image_id"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df["image_id"] == image_id]
        image = self.read_image(os.path.join(self.image_dir, str(image_id)))
        image = image / 255.0

        boxes = records[["x", "y", "w", "h"]].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max = x_min(x) + w
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max = y_min(y) + h

        area = records["box_area"].values
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            sample = {
                "image": image,
                "bboxes": target["boxes"],
                "labels": labels,
            }
            sample = self.transforms(**sample)
            image = sample["image"]

            target["boxes"] = torch.stack(
                tuple(map(torch.tensor, zip(*sample["bboxes"])))
            ).permute(1, 0)

        return image.float(), target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def read_image(self, path):
        image = cv2.imread(path + ".jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        return image
