import os
import logging
from pathlib import Path
from typing import List

import cv2
import selfies
import torch
import numpy as np

from innofw.constants import Frameworks, Stages
from innofw.core.datasets.hdf5 import HDF5Dataset
from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)


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
            w_sampler=False,
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

        self.aug = {'train': None, 'test': None, 'val': None} if augmentations is None else augmentations
        self.channels_num = channels_num
        self.val_size = val_size
        self.random_seed = random_seed
        self.w_sampler = w_sampler

        self.mul = 1

    def setup_train_test_val(self, **kwargs):
        # files = list(self.data_path.rglob(''))
        train_files = self.find_hdf5(self.train_dataset)
        test_files = self.find_hdf5(self.test_dataset)

        self.random_split = False

        # prepare datasets
        if self.random_split:
            train_val = HDF5Dataset(train_files, self.channels_num, self.aug['train'])
            val_size = int(len(train_val) * float(self.val_size))
            train, val = torch.utils.data.random_split(
                train_val, [len(train_val) - val_size, val_size]
            )

            self.train_ds = train
            self.val_ds = val
            # Set validatoin augmentations for val
            setattr(self.val_ds, 'transform', self.aug['val'])
        else:
            val_files = [f for f in train_files if 'val' in Path(f).name]
            train_files = [f for f in train_files if 'train' in Path(f).name]

            from innofw.core.datasets.segmentation_hdf5_old_pipe import DatasetUnion, Dataset
            self.val_ds = DatasetUnion([Dataset(path_to_hdf5=f, in_channels=self.channels_num, augmentations=self.aug['val']) for f in val_files])  # HDF5Dataset(val_files, self.channels_num, self.aug['val'])
            self.train_ds = DatasetUnion([Dataset(path_to_hdf5=f, in_channels=self.channels_num, augmentations=self.aug['train']) for f in train_files])  # HDF5Dataset(train_files, self.channels_num, self.aug['train'])

        # Applly test transform
        self.test_ds = None
        # self.test_ds = HDF5Dataset(test_files, self.channels_num, self.aug['test'])

    def find_hdf5(self, path: Path) -> List[Path]:
        paths = []
        if not os.path.isfile(path):
            for p in os.listdir(path):
                paths.append(os.path.join(path, p))
        return paths or [path]

    def setup_infer(self):
        pass

    def train_dataloader(self):
        class_weights = [w if id_ != 1 else w * self.mul for w, id_ in zip(self.train_ds.class_weights, self.train_ds.class_ids)]
        return torch.utils.data.DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                sampler=torch.utils.data.WeightedRandomSampler(class_weights, 
                                                            len(self.train_ds)) if self.w_sampler else None,
                drop_last=True,
                num_workers=self.num_workers,
                # shuffle=True, # unsupported with 'sampler' argument
                worker_init_fn=lambda _: np.random.seed()
            )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            sampler=torch.utils.data.WeightedRandomSampler(self.val_ds.class_weights, len(self.val_ds)),
            num_workers=self.num_workers,
            drop_last=True,
    #         worker_init_fn=lambda _: np.random.seed()
        )

    # test_dataloader?

    def setup_infer(self):
        if isinstance(self.predict_dataset, HDF5Dataset):
            return
        infer_files = self.find_hdf5(self.predict_ds)
        self.predict_dataset = HDF5Dataset(infer_files, self.channels_num, self.aug['test'])

    def save_preds(self, preds, stage: Stages, dst_path: Path):
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
