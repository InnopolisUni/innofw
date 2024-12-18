import os
import pathlib

import albumentations as albu
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from innofw.constants import Stages
from innofw.core.augmentations import Augmentation
from innofw.core.datamodules.lightning_datamodules.base import BaseLightningDataModule
from innofw.core.datasets.coco_rtk import DicomCocoDatasetRTK


class CustomNormalize:
    def __call__(self, image, **kwargs):
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image


class DicomCocoComplexingDataModule(BaseLightningDataModule):
    task = ["image-detection", "image-segmentation"]
    dataset = DicomCocoDatasetRTK

    def __init__(
        self,
        train=None,
        test=None,
        infer=None,
        val_size: float = 0.2,
        num_workers: int = 1,
        augmentations=None,
        stage=None,
        batch_size=32,
        transform=None,
        val_split=0.2,
        test_split=0.1,
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

    def setup(self, stage=None):
        pass

    def setup_train_test_val(self, **kwargs):
        pass

    def setup_infer(self):
        if self.aug:
            transform = Augmentation(self.aug["test"])
        else:

            transform = albu.Compose(
                [
                    albu.Resize(256, 256),
                    albu.Lambda(image=CustomNormalize()),
                    ToTensorV2(transpose_mask=True),
                ]
            )
        if str(self.predict_source).split("/")[-1] in ["mrt", "ct"]:
            self.predict_source = self.predict_source.parent
        cont = os.listdir(self.predict_source)
        assert "ct" in cont, f"No CT data in {self.predict_source}"
        assert "mrt" in cont, f"No MRT data in {self.predict_source}"

        self.predict_dataset = [
            self.dataset(
                data_dir=os.path.join(self.predict_source, "ct"),
                transform=transform,
            ),
            self.dataset(
                data_dir=os.path.join(self.predict_source, "mrt"),
                transform=transform,
            ),
        ]
        self.predict_dataset = torch.utils.data.ConcatDataset(self.predict_dataset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """shuffle should be turned off"""
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        """we assume that shuffle is turned off

        Args:
            preds:
            stage:
            dst_path:

        Returns:

        """

        total_iter = 0
        for tensor_batch in preds:
            for i in range(tensor_batch.shape[0]):
                path = self.predict_dataset[total_iter]["path"]
                output = tensor_batch[i].cpu().detach().numpy()
                output = np.max(output, axis=0)
                output = np.expand_dims(output, axis=0)
                output = np.transpose(output, (1, 2, 0))
                if "/ct/" in path:
                    prefix = "_ct"
                else:
                    prefix = "_mrt"
                path = os.path.join(dst_path, f"{prefix}_{total_iter}.npy")
                np.save(path, output)
                total_iter += 1


class DicomCocoDataModuleRTK(DicomCocoComplexingDataModule):
    def setup_infer(self):
        if self.aug:
            transform = Augmentation(self.aug["test"])
        else:

            transform = albu.Compose(
                [
                    albu.Resize(256, 256),
                    albu.Lambda(image=CustomNormalize()),
                    ToTensorV2(transpose_mask=True),
                ]
            )
        self.predict_dataset = self.dataset(
            data_dir=str(self.predict_source), transform=transform
        )

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
