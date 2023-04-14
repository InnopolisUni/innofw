
from torch.utils.data import DataLoader
from innofw.constants import Frameworks
from innofw.core.datamodules.lightning_datamodules.base import BaseLightningDataModule
from pydantic import validate_arguments
from torch.utils.data import ConcatDataset
from innofw.constants import Frameworks, Stages
from innofw.core.augmentations import Augmentation
import logging
import os.path
import pathlib
import pandas as pd
from innofw.core.datamodules.lightning_datamodules.drugprot import DataCollatorWithPaddingAndTruncation
from torch.utils.data import WeightedRandomSampler
import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))

class ConcatenatedLightningDatamodule(BaseLightningDataModule):

    framework = [Frameworks.torch]

    @validate_arguments
    def __init__(
        self,
        class_name,
        datamodules,
        batch_size: int = 4,
        val_size: float = 0.2,
        num_workers: int = 1,
        augmentations=None,
        infer=None,
        stage=None,
        channels_num = None,
        random_seed = None,
        w_sampler = False,
        *args,
        **kwargs
    ):
        self.datamodules = datamodules 
        self.batch_size = batch_size
        self.shuffle = True
        self.num_workers = num_workers
        self.stage = stage
        self.infer = infer
        self.val_size = val_size
        self.augmentations=augmentations
        self.class_name = class_name
        self.channels_num = channels_num
        self.val_size = val_size
        self.random_seed = random_seed
        self.w_sampler = w_sampler
        self.mul = 1
        self.batch_sampler = None
        self.collate_fn = None
        self.pin_memory = False
        self.worker_init_fn = None
        self.prefetch_factor = 1
        self.persistent_workers = False
        self.timeout = 0
        self.shuffle = False
        self.drop_last = False
        
    def setup_train_test_val(self, **kwargs):
        [dm.setup_train_test_val() for dm in self.datamodules]
        self.train_ds = ConcatDataset([dm.train_dataset for dm in self.datamodules])
        #self.test_ds = ConcatDataset([dm.test_dataset for dm in self.datamodules])
        self.val_ds = ConcatDataset([dm.val_dataset for dm in self.datamodules])

    def stage_dataloader(self, dataset, stage):
        if stage in ["val", "test", "predict"]:
            self.shuffle = False
            self.drop_last = False
        else:
            self.shuffle = self.shuffle
            self.drop_last = True
            
        if self.class_name == "CocoLightningDataModule":
           self.collate_fn = collate_fn
        elif self.class_name == "DrugprotDataModule":
            self.collate_fn = DataCollatorWithPaddingAndTruncation(
            max_length=512,
            sequence_keys=["input_ids", "labels"]
        )   
        elif self.class_name == "ImageLightningDataModule":
            self.prefetch_factor = 2
        elif self.class_name == "SegmentationDM":
            self.prefetch_factor = 2
            self.pin_memory = True
            self.prefetch_factor = 2
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            #sampler=sampler,
            #batch_sampler=self.batch_sampler,
        )       

    def train_dataloader(self):
        return self.stage_dataloader(self.train_ds, "train")

    def val_dataloader(self):
        return self.stage_dataloader(self.val_ds, "val")

    def test_dataloader(self):
        return self.stage_dataloader(self.test_ds, "test")

    def predict_dataloader(self):
        return self.stage_dataloader(self.predict_source, "predict")
    
    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        pass