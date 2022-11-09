import logging
import os
import pathlib

import h5py
import torch
import rasterio as rio

from innofw.constants import Frameworks, Stages

from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)
from innofw.core.datasets.hdf5 import HDF5Dataset
from innofw.core.datasets.rasterio import RasterioDataset


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

        # TODO: should object instantiation be here?
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
        train_val = HDF5Dataset(train_files, self.channels_num, self.aug)
        val_size = int(len(train_val) * float(self.val_size))
        train, val = torch.utils.data.random_split(
            train_val, [len(train_val) - val_size, val_size]
        )

        self.train_dataset = train
        self.test_dataset = HDF5Dataset(test_files, self.channels_num, self.aug)
        self.val_dataset = val

    def setup_infer(self):
        if isinstance(self.predict_dataset, HDF5Dataset):
            return
        infer_files = self.find_hdf5(self.predict_dataset)
        self.predict_dataset = HDF5Dataset(infer_files, self.channels_num, self.aug)

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        dataloader = self.get_stage_dataloader(stage)
        out_file_path = dst_path / "results"
        os.mkdir(out_file_path)
        filename = out_file_path / "out.hdf5"
        with h5py.File(filename, 'w') as f:
            len_ = len(preds)
            f.create_dataset('len', data=[len_])
            for preds_batch in preds:
                for i , pred in enumerate(preds_batch):
                    pred = pred.numpy()
                    pred[pred < 0.3] = 0
                    f.create_dataset(str(i), data=pred)

        logging.info(f"Saved result to: {out_file_path}")
