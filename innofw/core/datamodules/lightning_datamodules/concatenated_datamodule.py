
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

def collate_fn(batch):
    return tuple(zip(*batch))
class ConcatenatedLightningDatamodule(BaseLightningDataModule):

    framework = [Frameworks.torch]

    @validate_arguments
    def __init__(
        self,
        datamodules,
        batch_size: int = 16,
        val_size: float = 0.2,
        num_workers: int = 1,
        augmentations=None,
        infer=None,
        stage=None,
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

    def setup_train_test_val(self, **kwargs):
        [dm.setup_train_test_val() for dm in self.datamodules]
        #self.save_preds = self.datamodules[0].save_preds
        self.train_ds = ConcatDataset([dm.train_dataset for dm in self.datamodules])
        self.test_ds = ConcatDataset([dm.test_dataset for dm in self.datamodules])
        self.val_ds = ConcatDataset([dm.val_dataset for dm in self.datamodules])

    def stage_dataloader(self, dataset, stage):
        if stage in ["val", "test", "predict"]:
            shuffle = False
            drop_last = False
        else:
            shuffle = self.shuffle
            drop_last = True
        # detection_coco
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

        # sampler = self.samplers[stage]

        # if sampler is not None:
        #     shuffle = False
        
        # image_folder_dm
        # return DataLoader(
        #     dataset,
        #     batch_size=self.batch_size,
        #     shuffle=shuffle,
        #     #sampler=sampler,
        #     batch_sampler=None,
        #     num_workers=self.num_workers,
        #     collate_fn=None,
        #     pin_memory=False,  # True
        #     drop_last=drop_last,
        #     timeout=0,  # pin_memory=False
        #     worker_init_fn=None,
        #     prefetch_factor=2,
        #     persistent_workers=False,
        # )
        
        

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
        # out = []
        # for sublist in preds:
        #     out.extend(sublist.tolist())
        # images = self.predict_source.image_names
        # df = pd.DataFrame(list(zip(images, out)), columns=["Image name", "Class"])
        # dst_filepath = os.path.join(dst_path, "classification.csv")
        # df.to_csv(dst_filepath)
        # logging.info(f"Saved results to: {dst_filepath}")