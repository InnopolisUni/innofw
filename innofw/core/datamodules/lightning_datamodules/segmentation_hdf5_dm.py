import logging
import os
import pathlib

import cv2
import torch
import numpy as np

from innofw.constants import Frameworks, Stages

from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)
from innofw.core.datasets.hdf5 import HDF5Dataset


class HDF5LightningDataModule(BaseLightningDataModule):
    """Class defines hdf5 dataset preparation and dataloader creation for semantic segmentation

        Attributes
        ----------
        task: List[str]
            the task the datamodule is intended to be used for
        framework: List[Union[str, Frameworks]]
            the model framework the datamodule is designed to work with

        Methods
        -------
        setup_train_test_val
            finds hdf5 files
            splits train data into train and validation sets
            creates dataset objects
        save_preds(preds, stage: Stages, dst_path: pathlib.Path)
            saves predicted segmentation masks as file in the destination folder
    """
    task = ["image-segmentation"]
    framework = [Frameworks.torch]

    def __init__(
            self,
            train,
            test,
            infer=None,
            augmentations=None,
            channels_num: int = 3,
            val_size: float = 0.2,
            batch_size: int = 32,
            num_workers: int = 1,
            random_seed: int = 42,
            stage=None,
            *args,
            **kwargs,
    ):
        super().__init__(
            train=train,
            test=test,
            batch_size=batch_size,
            num_workers=num_workers,
            infer=infer,
            stage=stage,
            *args,
            **kwargs,
        )

        self.aug = augmentations
        self.channels_num = channels_num
        self.val_size = val_size
        self.random_seed = random_seed

    def find_hdf5(self, path):
        paths = []
        if not os.path.isfile(path):
            for p in os.listdir(path):
                paths.append(os.path.join(path, p))
        return paths or [path]

    def setup_train_test_val(self, **kwargs):
        # files = list(self.data_path.rglob(''))
        train_files = self.find_hdf5(self.train_dataset)
        test_files = self.find_hdf5(self.test_dataset)

        # prepare datasets
        train_val = HDF5Dataset(train_files, self.channels_num, self.aug['train'])
        val_size = int(len(train_val) * float(self.val_size))
        train, val = torch.utils.data.random_split(
            train_val, [len(train_val) - val_size, val_size]
        )

        self.train_dataset = train
        self.val_dataset = val
        # Set validatoin augmentations for val
        setattr(self.val_dataset, 'transform', self.aug['val'])
        # Applly test transform
        self.test_dataset = HDF5Dataset(test_files, self.channels_num, self.aug['test'])

    def setup_infer(self):
        if isinstance(self.predict_dataset, HDF5Dataset):
            return
        infer_files = self.find_hdf5(self.predict_dataset)
        self.predict_dataset = HDF5Dataset(infer_files, self.channels_num, self.aug['test'])

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        out_file_path = dst_path / "results"
        os.mkdir(out_file_path)
        for preds_batch in preds:
            for i , pred in enumerate(preds_batch):
                pred = pred.numpy()
                pred[pred < 0.3] = 0
                pred[pred > 0] = 255
                filename = out_file_path / f"out_{i}.png"
                cv2.imwrite(filename, pred[0])
        logging.info(f"Saved result to: {out_file_path}")
