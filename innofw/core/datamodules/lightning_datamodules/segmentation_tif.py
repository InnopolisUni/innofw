from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pydantic import DirectoryPath
from pydantic import FilePath
from pydantic import validate_arguments
from segmentation.datamodules.datasets.roads_dataset import SegmentationDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

#
#


class SegmentationDM(pl.LightningDataModule):
    @validate_arguments
    def __init__(
        self,
        img_path: DirectoryPath,
        label_path: DirectoryPath,
        weights_csv_path: Optional[FilePath] = None,
        train_transform=None,
        val_transform=None,
        val_size=0.2,
        batch_size=8,
        channels: Optional[int] = None,
        num_workers: int = 8,
        shuffle=True,
    ):
        super().__init__()
        self.img_path = img_path
        self.label_path = label_path

        if weights_csv_path:
            self.weights = pd.read_csv(weights_csv_path)
            assert "file_names" in self.weights and "weights" in self.weights
        else:
            self.weights = None

        self.train_transform = train_transform
        self.val_transform = val_transform

        self.channels = channels

        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.random_seed = 42

    def setup(self, stage, **kwargs):
        if self.weights is not None:
            images = [
                self.img_path / img_name
                for img_name in self.weights["file_names"]
            ]
            masks = [
                self.label_path / img_name
                for img_name in self.weights["file_names"]
            ]
            train_weights, val_weights = train_test_split(
                self.weights["weights"],
                test_size=self.val_size,
                random_state=self.random_seed,
            )
            train_sampler = WeightedRandomSampler(
                torch.tensor(list(train_weights)), len(train_weights)
            )
            val_sampler = WeightedRandomSampler(
                torch.tensor(list(val_weights)), len(val_weights)
            )
            self.samplers = {"train": train_sampler, "val": val_sampler}
        else:
            images = list(self.img_path.rglob("*.tif"))
            masks = list(self.label_path.rglob("*.tif"))
            images = sorted(images, key=lambda x: x.name)
            masks = sorted(masks, key=lambda x: x.name)
            self.samplers = {"train": None, "val": None}

        assert len(images) == len(
            masks
        ), "number of images and masks should be equal"

        train_images, val_images, train_masks, val_masks = train_test_split(
            images,
            masks,
            test_size=self.val_size,
            random_state=self.random_seed,
        )

        self.train_dataset = SegmentationDataset(
            train_images,
            train_masks,
            transform=self.train_transform,
            channels=self.channels,
        )
        self.val_dataset = SegmentationDataset(
            val_images,
            val_masks,
            transform=self.val_transform,
            channels=self.channels,
        )

    def stage_dataloader(self, dataset, stage):
        if stage in ["val", "test", "predict"]:
            shuffle = False
            drop_last = False
        else:
            shuffle = self.shuffle
            drop_last = True

        sampler = self.samplers[stage]

        if sampler is not None:
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=None,
            num_workers=self.num_workers,
            collate_fn=None,
            pin_memory=True,
            drop_last=drop_last,
            timeout=0,  # pin_memory=False
            worker_init_fn=None,
            prefetch_factor=2,
            persistent_workers=False,
        )

    def train_dataloader(self):
        return self.stage_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self.stage_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self.stage_dataloader(self.val_dataset, "val")

    def predict_dataloader(self):
        return self.stage_dataloader(self.predict_dataset, "predict")


if __name__ == "__main__":
    from segmentation.utils.config import read_cfg

    dm = read_cfg(
        "/home/qazybek/repos/segmentation/config/datamodules/100123_roads_bin_seg.yaml"
    )
    dm.setup("train")
    dl = dm.train_dataloader()
    batch = next(iter(dl))
