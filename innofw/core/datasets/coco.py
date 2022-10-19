import os
from pathlib import Path

import cv2
import numpy as np
import torch
from innofw.utils.data_utils.preprocessing.dicom_handler import dicom_to_img
from torch.utils.data import Dataset


class CocoDataset(Dataset):
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
            sample = {"image": image, "bboxes": target["boxes"], "labels": labels}
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


class DicomCocoDataset(CocoDataset):
    def read_image(self, path):
        return dicom_to_img(path + ".dcm")


class DicomCocoDatasetInfer(Dataset):
    def __init__(self, dicom_dir, transforms=None):
        self.images = []
        self.paths = [os.path.join(dicom_dir, d) for d in os.listdir(dicom_dir)]
        for dicom in self.paths:
            self.images.append(transforms(image=dicom_to_img(dicom))["image"])

    def __getitem__(self, idx):
        return self.images[idx]

    def __len__(self):
        return len(self.images)


class WheatDataset(Dataset):
    """A dataset example for GWC 2021 competition."""

    def __init__(self, annotations, root_dir, transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional data augmentation to be applied
                on a sample.
        """

        self.root_dir = Path(root_dir)
        self.image_list = annotations["image_name"].values
        self.domain_list = annotations["domain"].values
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        imgp = str(self.root_dir / (self.image_list[idx] + ".png"))
        domain = self.domain_list[
            idx
        ]  # We don't use the domain information but you could !
        bboxes = self.boxes[idx]
        img = cv2.imread(imgp)
        image = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # Opencv open images in BGR mode by default

        if self.transforms:
            transformed = self.transforms(
                image=image, bboxes=bboxes, class_labels=["wheat_head"] * len(bboxes)
            )  # Albumentations can transform images and boxes
            image = transformed["image"]
            bboxes = transformed["bboxes"]

        if len(bboxes) > 0:
            bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
            bboxes = torch.zeros((0, 4))
        return image, bboxes, domain

    def decodeString(self, BoxesString):
        """
        Small method to decode the BoxesString
        """
        if BoxesString == "no_box":
            return np.zeros((0, 4))
        else:
            try:
                boxes = np.array(
                    [
                        np.array([int(i) for i in box.split(" ")])
                        for box in BoxesString.split(";")
                    ]
                )
                return boxes
            except:
                print(BoxesString)
                print("Submission is not well formatted. empty boxes will be returned")
                return np.zeros((0, 4))
