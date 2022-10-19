import logging
import pathlib

import pandas as pd

from innofw.constants import Frameworks, Stages
from innofw.core.augmentations import Augmentation
from innofw.core.datasets.siamese_dataset import (
    SiameseDataset,
    SiameseDatasetInfer,
)
from torch.utils.data import random_split
import albumentations as albu
import albumentations.pytorch as albu_pytorch

from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)


class SiameseDataModule(BaseLightningDataModule):
    task = ["one-shot-learning"]
    framework = [Frameworks.torch]

    def __init__(
        # todo: add types to train and test parameters
        # todo: add type to the aug parameter
        self,
        train,
        test,
        infer=None,
        batch_size: int = 8,
        val_size: float = 0.2,
        num_workers: int = 1,
        augmentations=None,
        stage=None,
        *args,
        **kwargs,
    ):
        super().__init__(train, test, infer, batch_size, num_workers, stage)
        self.aug = augmentations
        self.val_size = val_size

    def setup_train_test_val(self, **kwargs):
        if self.aug:
            train_dataset = SiameseDataset(str(self.train_dataset), self.aug)
            self.test_dataset = SiameseDataset(str(self.test_dataset), self.aug)
        else:
            train_dataset = SiameseDataset(
                str(self.train_dataset),
                transform=Augmentation(
                    albu.Compose([albu_pytorch.transforms.ToTensorV2()])
                ),
            )
            self.test_dataset = SiameseDataset(
                str(self.test_dataset),
                transform=Augmentation(
                    albu.Compose([albu_pytorch.transforms.ToTensorV2()])
                ),
            )

        # divide into train, val, test
        n = len(train_dataset)
        train_size = int(n * (1 - self.val_size))
        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [train_size, n - train_size]
        )

    def setup_infer(self):
        if self.aug:
            self.predict_dataset = SiameseDatasetInfer(
                str(self.infer),
                self.aug,
            )
        else:
            self.predict_dataset = (
                SiameseDatasetInfer(
                    str(self.infer),
                    transform=Augmentation(
                        albu.Compose([albu_pytorch.transforms.ToTensorV2()])
                    ),
                ),
            )

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        images = self.predict_dataset.image_pair_names
        df = pd.DataFrame(list(zip(images, preds)), columns=["Image name", "Class"])
        dst_filepath = pathlib.Path(dst_path) / "preds.csv"  # todo: should it be fixed?
        df.to_csv(dst_filepath)
        logging.info(f"Saved results to: {dst_filepath}")
