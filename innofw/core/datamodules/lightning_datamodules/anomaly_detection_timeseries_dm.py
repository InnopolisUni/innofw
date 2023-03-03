import pathlib

import logging
import pandas as pd
import torch
from torch.utils.data import random_split

from innofw.constants import Stages, Frameworks
from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)
from innofw.utils.dm_utils.utils import find_file_by_ext
from innofw.core.datasets.timeseries import ECGDataset


def collate_fn(batch):
    return tuple(zip(*batch))


class TimeSeriesLightningDataModule(BaseLightningDataModule):
    """
        A Class used for working with Time Series
        ...

        Attributes
        ----------
        aug : dict
            The list of augmentations
        val_size: float
            The proportion of the dataset to include in the validation set

        Methods
        -------
        save_preds(preds, stage: Stages, dst_path: pathlib.Path):
            Saves inference predictions to csv file

        setup_infer():
            The method prepares inference data

    """

    task = ["anomaly-detection-timeseries"]
    framework = [Frameworks.torch]

    def __init__(
        self,
        train,
        test,
        infer=None,
        batch_size: int = 2,
        val_size: float = 0.2,
        num_workers: int = 1,
        augmentations=None,
        stage=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            train, test, infer, batch_size, num_workers, stage, *args, **kwargs
        )
        self.aug = augmentations
        self.val_size = val_size

    def setup_train_test_val(self, **kwargs):
        train_dataset = ECGDataset(find_file_by_ext(self.train_source, ".csv"))
        self.test_dataset = ECGDataset(find_file_by_ext(self.test_source, ".csv"))

        # divide into train, val, test
        n = len(train_dataset)
        train_size = int(n * (1 - self.val_size))
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [train_size, n - train_size]
        )

    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # collate_fn=collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # collate_fn=collate_fn,
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

    def predict_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
        return test_dataloader

    def setup_infer(self):
        self.predict_dataset = ECGDataset(find_file_by_ext(self.infer, ".csv"))

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        dst_path = pathlib.Path(dst_path)
        df = pd.DataFrame(list(preds), columns=["prediction"])
        dst_filepath = dst_path / "prediction.csv"
        df.to_csv(dst_filepath)
        logging.info(f"Saved results to: {dst_filepath}")
