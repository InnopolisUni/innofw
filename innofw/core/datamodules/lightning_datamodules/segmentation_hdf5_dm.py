import logging
import os
import pathlib

import torch
import rasterio as rio

from innofw.constants import Frameworks, Stages

from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)
from innofw.core.datasets.hdf5 import HDF5Dataset
from innofw.core.datasets.rasterio import RasterioDataset


class HDF5LightningDataModule(BaseLightningDataModule):
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
        if self.predict_dataset.is_file():
            predict_files = [self.predict_dataset]
        else:
            predict_files = list(self.predict_dataset.iterdir())

        self.predict_dataset = RasterioDataset(
            raster_files=predict_files,
            bands_num=self.channels_num,
            mask_files=None,
            transform=self.aug,
        )

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        dataloader = self.get_stage_dataloader(stage)
        for pred, batch in zip(preds, dataloader):
            for file, output in zip(batch["names"], pred["preds"]):
                file_path = pathlib.Path(file)
                out = output.squeeze()

                with rio.open(file_path, "r") as f:
                    # retrieve metadata from source rasters
                    meta = f.meta
                    meta.update(
                        height=out.shape[0],
                        width=out.shape[1],
                    )
                    out_file_path = dst_path / "results" / file_path.name
                    out_file_path.parent.mkdir(exist_ok=True, parents=True)
                    # save the result
                    with rio.open(out_file_path, "w+", **meta) as f_out:
                        f_out.write(out, 1)

                    logging.info(f"Saved result to: {out_file_path}")
