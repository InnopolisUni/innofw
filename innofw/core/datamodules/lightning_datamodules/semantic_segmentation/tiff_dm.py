import logging
from typing import Optional, Union, List, Tuple
from omegaconf.listconfig import ListConfig
from pathlib import Path
from functools import reduce

#
import pandas as pd
import pytorch_lightning as pl
from pydantic import validate_arguments, FilePath, DirectoryPath
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch

#
from innofw.core.datasets.semantic_segmentation.tiff_dataset import SegmentationDataset


def get_samples(path) -> List[Path]:  # todo: move out
    if isinstance(path, list):
        samples = reduce(lambda x, y: x + y, [list(p.rglob("*.tif")) for p in path])
    else:
        samples = list(path.rglob("*.tif"))

    samples = sorted(samples, key=lambda x: f"{x.parent.name}{x.name}")
    return samples

from innofw.constants import Frameworks

class SegmentationDM(pl.LightningDataModule):
    task = ["image-segmentation"]
    framework = [Frameworks.torch]

    @validate_arguments
    def __init__(
        self,
        img_path,
        label_path,  # : Union[DirectoryPath, ListConfig]
        weights_csv_path: Optional[FilePath] = None,
        filtered_files_csv_path: Optional[FilePath] = None,
        train_transform=None,
        stage=None,
        augmentations=None,
        val_transform=None,
        val_size=0.2,
        batch_size=8,
        channels: Optional[int] = None,
        num_workers: int = 8,
        shuffle=True,
        with_caching: bool = False,
    ):
        super().__init__()
        self.img_path = [Path(p) for p in img_path] if isinstance(img_path, ListConfig) else Path(img_path)
        self.label_path = [Path(p) for p in label_path] if isinstance(label_path, ListConfig) else Path(label_path)

        self.weights = None

        if weights_csv_path:
            self.weights = pd.read_csv(weights_csv_path)
            assert "file_names" in self.weights and "weights" in self.weights

        self.filtered_files = None
        # convert it to the flag
        if filtered_files_csv_path is not None:  # tod: convert csv generator to a script from notebook
            self.filtered_files = pd.read_csv(filtered_files_csv_path)
            assert 'filename' in self.filtered_files
            self.filtered_files = self.filtered_files.set_index('filename').index

            if self.weights is not None:
                logging.info("filtering samples and weights using csv file")
                i2 = self.weights.set_index('file_names').index
                # filter values
                self.weights = self.weights[i2.isin(self.filtered_files)]

        self.train_transform = train_transform
        self.val_transform = val_transform

        self.channels = channels

        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.random_seed = 42
        self.with_caching = with_caching

    def teardown(self, stage: str):
        # delete files generated at the stage prepare_data
        pass

    def prepare_data(self):
        # convert geometry files(e.g. .shp) to masks(e.g. .tif)
        # add an ability to check if such files are already created

        # if link provided as a data source then download and process here
        """
            datamodule_conf:
                img_path:
                label_path:
                batch_size:

                data_prep:
                    reproject
                    rasterize
                    cut_by_aoi
                    cut_to_tiles
                    save_to_disk
        """
        # btw: It is not recommended to assign state here(ref: https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html)
        pass 

    # todo: add datamodule checkpointing

    def setup(self, stage, **kwargs):
        if self.weights is None:
            images = get_samples(self.img_path)
            masks = get_samples(self.label_path)

            if self.filtered_files is not None:
                logging.info("filtering samples using csv file")
                # filter
                def filter_samples(samples, filtered_names):
                    f_samples = [s for s in samples if s.name in filtered_names]
                    assert len(f_samples) == len(filtered_names)  # todo: not always will be true
                    return f_samples

                images = filter_samples(images, self.filtered_files)
                masks = filter_samples(masks, self.filtered_files)

            self.samplers = {"train": None, "val": None}
        else:
            images = [
                self.img_path / img_name for img_name in self.weights["file_names"]
            ]
            masks = [
                self.label_path / img_name for img_name in self.weights["file_names"]
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

        print(len(images), len(masks))
        assert len(images) == len(masks), "number of images and masks should be equal"

        train_images, val_images, train_masks, val_masks = train_test_split(
            images, masks, test_size=self.val_size, random_state=self.random_seed
        )

        self.train_dataset = SegmentationDataset(
            train_images,
            train_masks,
            transform=self.train_transform,
            channels=self.channels,
            with_caching=self.with_caching
        )
        self.val_dataset = SegmentationDataset(
            val_images, val_masks, transform=self.val_transform, channels=self.channels, with_caching=self.with_caching
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
            pin_memory=False,  # True
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
    from innofw.utils.config import read_cfg

    dm = read_cfg(Path(
        "/home/qazybek/repos/segmentation/config/datamodules/100123_roads_bin_seg.yaml"
    ))
    dm.setup("train")
    dl = dm.train_dataloader()
    batch = next(iter(dl))
