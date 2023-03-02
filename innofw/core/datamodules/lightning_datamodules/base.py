from abc import ABC
from typing import List, Union

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import albumentations as albu
from albumentations.pytorch import ToTensorV2
import albumentations.pytorch as albu_pytorch

from innofw.core.datamodules.base import BaseDataModule
from innofw.core.augmentations import Augmentation
from innofw.core.datasets.image_infer import ImageFolderInferDataset
from innofw.constants import Frameworks


class BaseLightningDataModule(BaseDataModule, pl.LightningDataModule, ABC):
    """
        An abstract class to define interface and methods of datamodules for the torch framework

        Attributes
        ----------
        framework: List[Union[str, Frameworks]]
            the model framework the datamodule is designed to work with

        Methods
        -------
        train_dataloader()
            Returns torch.utils.data.Dataloader using the training dataset

        predict_dataloader()
            Returns torch.utils.data.Dataloader using the inference dataset
    """
    framework: List[Union[str, Frameworks]] = [Frameworks.torch]

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
        self.train_source = self.train
        self.test_source = self.test
        self.predict_source = self.infer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_source = None
        self.aug = None

        # self.train_dataset  # make it abstract property?

    def get_aug(self, all_augmentations, stage):
        if self.aug is not None and stage in all_augmentations and all_augmentations[stage] is not None:
            return Augmentation(all_augmentations[stage])
        return Augmentation(
                albu.Compose([albu_pytorch.transforms.ToTensorV2()])
        )

    def prepare_data(self):
        pass

    def setup_infer(self):
        if self.aug:
            self.predict_dataset = ImageFolderInferDataset(
                str(self.predict_source),
                transforms=Augmentation(self.aug['test']),
            )
        else:
            self.predict_dataset = ImageFolderInferDataset(
                str(self.predict_source),
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
