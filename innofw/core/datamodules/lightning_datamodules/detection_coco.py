import os
import pathlib

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import random_split

from innofw.constants import Stages
from innofw.core.augmentations import Augmentation
from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)
from innofw.core.datasets.coco import CocoDataset
from innofw.core.datasets.coco import DicomCocoDataset
from innofw.core.datasets.coco import DicomCocoDatasetInfer
from innofw.utils.data_utils.preprocessing.dicom_handler import dicom_to_img
from innofw.utils.data_utils.preprocessing.dicom_handler import img_to_dicom
from innofw.utils.dm_utils.utils import find_file_by_ext
from innofw.utils.dm_utils.utils import find_folder_with_images


def collate_fn(batch):
    return tuple(zip(*batch))


class CocoLightningDataModule(BaseLightningDataModule):
    """
    A Class used for working with data in COCO format
    ...

    Attributes
    ----------
    aug : dict
        The list of augmentations
    val_size: float
        The proportion of the dataset to include in the validation set

    Methods
    -------
    find_csv_and_data(path):
        Returns paths to csv file with bounding boxes and folder with images

    """

    task = ["image-detection"]
    dataset = CocoDataset

    def __init__(
        self,
        train,
        test,
        batch_size: int = 16,
        infer=None,
        val_size: float = 0.2,
        num_workers: int = 1,
        augmentations=None,
        stage=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            train,
            test,
            infer,
            batch_size,
            num_workers,
            stage,
            *args,
            **kwargs,
        )
        self.aug = (
            {"train": None, "test": None, "val": None}
            if augmentations is None
            else augmentations
        )
        self.val_size = val_size

    def setup_train_test_val(self, **kwargs):
        self.train_source, train_csv = self.find_csv_and_data(self.train_source)
        self.test_source, test_csv = self.find_csv_and_data(self.test_source)
        self.aug = {"train": None, "test": None, "val": None}  # todo: fix
        if (
            self.aug is not None
            and self.aug["train"] is not None
            and self.aug["test"] is not None
        ):
            train_dataset = self.dataset(
                train_csv,
                str(self.train_source),
                # transforms=Augmentation(self.aug['train']),
            )
            self.test_dataset = self.dataset(
                test_csv,
                str(self.test_source),
                # transforms=Augmentation(self.aug['test']),
            )
        else:
            train_dataset = self.dataset(
                train_csv,
                str(self.train_source),
                transforms=albu.Compose(
                    [ToTensorV2(p=1.0)],
                    bbox_params={
                        "format": "pascal_voc",
                        "label_fields": ["labels"],
                    },
                ),
            )
            self.test_dataset = self.dataset(
                test_csv,
                str(self.test_source),
                transforms=albu.Compose(
                    [ToTensorV2(p=1.0)],
                    bbox_params={
                        "format": "pascal_voc",
                        "label_fields": ["labels"],
                    },
                ),
            )

        # divide into train, val, test
        n = len(train_dataset)
        train_size = int(n * (1 - self.val_size))
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [train_size, n - train_size]
        )
        # Set validatoin augmentations for val
        setattr(self.val_dataset, "transform", self.aug["val"])

    def find_csv_and_data(self, path):
        csv_path = find_file_by_ext(path, ".csv")
        train_df = pd.read_csv(csv_path)
        arr = train_df["bbox"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
        bboxes = np.stack(arr)
        for i, col in enumerate(["x", "y", "w", "h"]):
            train_df[col] = bboxes[:, i]
        train_df["box_area"] = train_df["w"] * train_df["h"]
        return find_folder_with_images(path), train_df

    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return test_dataloader

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        pass


class DicomCocoLightningDataModule(CocoLightningDataModule):
    dataset = DicomCocoDataset
    """
    A Class used for working with Dicom data in COCO format
    ...

    Attributes
    ----------

    Methods
    -------
    save_preds(preds, stage: Stages, dst_path: pathlib.Path):
        Saves inference predictions in Dicom format

    """

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        images = self.predict_dataset.images
        dicoms = self.predict_dataset.paths
        out = []
        for sublist in preds:
            out.extend(sublist)
        for p, dicom in zip(out, dicoms):
            boxes = p["boxes"].data.numpy()
            scores = p["scores"].data.numpy()
            boxes = boxes[scores >= 0.1].astype(np.int32)
            draw_boxes = boxes.copy()
            im_to_draw = dicom_to_img(dicom)
            for j, box in enumerate(draw_boxes):
                color = (255, 0, 0)
                cv2.rectangle(
                    im_to_draw,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    2,
                )
            img_to_dicom(
                im_to_draw,
                dicom,
                os.path.join(dst_path, dicom.split("/")[-1] + "SC"),
            )

    def setup_infer(self):
        transforms = (
            albu.Compose(
                [
                    albu.Normalize(
                        mean=0.5,
                        std=0.24,
                    ),
                    albu.ToGray(),
                    albu.Resize(512, 512),
                    ToTensorV2(p=1.0),
                ]
            ),
        )
        try:
            aug = self.aug["test"]
        except:
            aug = transforms
        self.predict_dataset = DicomCocoDatasetInfer(
            str(self.infer),
            Augmentation(aug),
        )
