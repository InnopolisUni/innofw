import albumentations
import numpy as np
import torch
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from innofw.core.augmentations import Augmentation
from innofw.core.datamodules.base import BaseDataModule
from abc import ABC
import albumentations as albu

from innofw.core.datasets.image_infer import ImageFolderInferDataset


class BaseLightningDataModule(BaseDataModule, pl.LightningDataModule, ABC):
    def __init__(
        self,
        train,
        test,
        infer=None,
        batch_size=1,
        num_workers=1,
        stage=None,
        *args,
        **kwargs
    ):
        super().__init__(train, test, infer, stage, *args, **kwargs)
        self.train_dataset = self.train  # todo: KA: we have to fix this mess.
        self.test_dataset = self.test
        self.predict_dataset = self.infer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_dataset = None
        self.aug = None

    def prepare_data(self):
        pass

    def setup_infer(self):
        if self.aug:
            self.predict_dataset = ImageFolderInferDataset(
                str(self.infer),
                transforms=Augmentation(self.aug),
            )
        else:
            self.predict_dataset = ImageFolderInferDataset(
                str(self.infer),
                transforms=Augmentation(
                    albu.Compose([ToTensorV2(p=1.0)]),
                ),
            )

    def _log_hyperparams(self):
        pass

    def prepare_data_per_node(self):
        pass

    def train_dataloader(self):
        train_dataloader = DataLoader(
            # todo: KA: I don't think it is an intuitive concept to use here: self.train_dataset
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return test_dataloader

    def predict_dataloader(self):
        pred_dataloader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return pred_dataloader
