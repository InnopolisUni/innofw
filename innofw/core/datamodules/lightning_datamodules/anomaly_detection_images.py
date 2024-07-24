import logging
import pathlib

import pandas as pd
import torch
from torch.utils.data import random_split

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)
from innofw.core.datasets.anomalies import AnomaliesDataset


class ImageAnomaliesLightningDataModule(BaseLightningDataModule):
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

    task = ["anomaly-detection-images"]
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
        self.train_dataset = AnomaliesDataset(self.train_source, self.aug, add_labels=False)
        self.test_dataset = AnomaliesDataset(self.test_source, add_labels=True)

        # divide into train, val, test - val is a part of test since train does not have anomalies
        n = len(self.test_dataset)
        test_size = int(n * (1 - self.val_size))
        self.test_dataset, self.val_dataset = random_split(
            self.test_dataset, [test_size, n - test_size]
        )

    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test_dataloader

    def predict_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test_dataloader

    def setup_infer(self):
        self.predict_dataset = AnomaliesDataset(self.predict_source)

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        dst_path = pathlib.Path(dst_path)
        df = pd.DataFrame(list(preds), columns=["prediction"])
        dst_filepath = dst_path / "prediction.csv"
        df.to_csv(dst_filepath)
        logging.info(f"Saved results to: {dst_filepath}")
