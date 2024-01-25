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
from innofw.core.datasets.coco import DicomCocoDataset_sm
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


import os
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class DicomCocoComplexingModule(pl.LightningDataModule):
    def __init__(self, train,
                 test,
                 infer=None,
                 val_size: float = 0.2,
                 num_workers: int = 1,
                 augmentations=None,
                 stage=None,
                 batch_size=32, transform=None, val_split=0.2, test_split=0.1,
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.val_split = val_split
        self.test_split = test_split
        self.infer = infer

    def setup(self, stage=None):
        pass

    def setup_infer(self):

        from albumentations import Compose
        from albumentations.pytorch.transforms import ToTensorV2
        from albumentations.augmentations import Normalize

        self.transform = Compose([ToTensorV2()])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        mrt_path = self.infer["target"]['mrt']
        ct_path = self.infer["target"]['ct']
        mrt_ds = DicomCocoDataset_sm(data_dir = mrt_path, transform=self.transform)
        ct_ds = DicomCocoDataset_sm(data_dir = ct_path, transform=self.transform)
        from torch.utils.data import ConcatDataset
        infer_ds = ConcatDataset(mrt_ds, ct_ds)
        return DataLoader(infer_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        pass


class DicomCocoDataModuleRTK(pl.LightningDataModule):
    def __init__(self, train,
                 test,
                 infer=None,
                 val_size: float = 0.2,
                 num_workers: int = 1,
                 augmentations=None,
                 stage=None,
                 batch_size=32, transform=None, val_split=0.2, test_split=0.1,
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.val_split = val_split
        self.test_split = test_split
        self.infer = infer

    def setup(self, stage=None):
        pass

    def setup_infer(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        infer_ds = DicomCocoDataset_sm(data_dir = self.infer["target"])
        return DataLoader(infer_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        prefix = "mask"
        for batch_idx, tensor_batch in enumerate(preds):
            for i in range(tensor_batch.shape[0]):
                output = tensor_batch[i].cpu().detach().numpy()
                output = np.max(output, axis=0)
                output = np.expand_dims(output, axis=0)
                output = np.transpose(output, (1, 2, 0))
                path = os.path.join(dst_path, f"{prefix}_{batch_idx}_{i}.npy")
                np.save(path, output)


# Функция для сохранения тензоров в виде масок
def save_tensor_list_as_masks3(tensor_list, prefix, dst_path):
    for tensor_idx, tensor in enumerate(tensor_list):
        for i in range(tensor.shape[0]):
            t = tensor[i].numpy()
            t = np.transpose(t, (1,2,0))
            path = os.path.join(dst_path, f"{prefix}_{tensor_idx}_{i}.npy")
            np.save(path, t)



            # # Для каждого канала делаем пороговое преобразование для создания маски
            # mask = tensor[i] > 0.5  # Порог можно настроить
            # # Преобразуем маску в numpy массив и сконвертируем в 8-битный формат
            # mask_np = (mask.numpy() * 255).astype(np.uint8)
            #
            # # Сохраняем каждый канал как отдельное изображение
            # for j in range(mask_np.shape[0]):
            #     img = Image.fromarray(mask_np[j])
            #     path = os.path.join(dst_path, f"{prefix}_{tensor_idx}_{i}_{j}.png")
            #     img.save(path)
