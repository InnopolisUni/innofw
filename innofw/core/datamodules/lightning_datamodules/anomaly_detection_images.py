import os
import logging
import pathlib

import pandas as pd
import torch
import cv2
import numpy as np
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
            val_size: float = 0.5,
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
        self.train_dataset = AnomaliesDataset(self.train_source, self.get_aug(self.aug, 'train'),
                                              add_labels=False)
        self.test_dataset = AnomaliesDataset(self.test_source, self.get_aug(self.aug, 'test'),
                                             add_labels=True)

        # divide into train, val, test - val is a part of test since train does not have anomalies
        n = len(self.test_dataset)
        test_size = int(n * (1 - self.val_size))
        self.test_dataset, self.val_dataset = random_split(
            self.test_dataset, [test_size, n - test_size]
        )

    def predict_dataloader(self):
        test_dataloader = torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test_dataloader

    def setup_infer(self):
        self.predict_dataset = AnomaliesDataset(self.predict_source, self.get_aug(self.aug, 'test'))

    def save_preds(self, out_batches, stage: Stages, dst_path: pathlib.Path):
        out_file_path = dst_path / "results"
        os.mkdir(out_file_path)
        n = 0
        for batch in out_batches:
            for img, pred in zip(batch[0], batch[1]):
                img = img.cpu().numpy()
                pred = pred.numpy() * 255  # shape - (1024, 1024)
                if pred.dtype != np.uint8:
                    pred = pred.astype(np.uint8)
                filename = out_file_path / f"out_{n}.png"
                n += 1
                cv2.imwrite(filename, pred)
                mask_vis = np.zeros_like(img)
                mask_vis[1, :, :] = pred / 255
                img_with_mask = (img * 255 * 0.75 + mask_vis * 255 * 0.25).astype(np.uint8).transpose((1, 2, 0))
                img_with_mask = cv2.cvtColor(img_with_mask, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(filename).replace('out_', 'vis_'), img_with_mask)
        logging.info(f"Saved result to: {out_file_path}")
