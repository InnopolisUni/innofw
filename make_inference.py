"""

Usage:
    >>> python make_inference.py models=deeplabv3_plus\
                                 datasets.source=/home/path/\
                                 ckpt_path=/home/path\
                                 datasets._target_=innofw.core.datamodules.TiledSegmentationDM

    >>> python make_inference.py models=deeplabv3_plus\
                                 ckpt_path=/home/path/\
                                 datasets.source=/some/folder\

    >>> python make_inference.py experiments=semantic-segmentation/KA_something\
                                 datasets.source=/home/path/\
                                 datasets.target=/home/other/path\
                                 ckpt_path=/home/path/\
                                 batch_size=8\
                                 accelerator=gpu\
"""
from typing import Any
from typing import List
from typing import Union

import hydra
from pydantic import HttpUrl
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer


class SegmentationLM(LightningModule):
    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx)


from abc import ABC, abstractmethod


class BaseSegmentationDM(LightningDataModule, ABC):
    """

    credit: https://lightning-flash.readthedocs.io/en/latest/api/generated/flash.image.instance_segmentation.data.InstanceSegmentationData.html#flash.image.instance_segmentation.data.InstanceSegmentationData
    """

    from innofw.constants import SegDataKeys, SegOutKeys

    DataKeys = SegDataKeys
    PredKeys = SegOutKeys

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def from_s3_data(
        self, predict_links: Union[List[HttpUrl], HttpUrl], batch_size
    ):
        ...

    @abstractmethod
    def from_folders(self, predict_folders):
        ...

    @abstractmethod
    def from_files(self, predict_files):
        ...


class SegmentationDM(BaseSegmentationDM):
    def __init__(self, predict_input):
        self.predict_input = predict_input

    def setup(self):
        self.predict_dataset = SomeDataset(self.predict_input)


class TiledMixin(ABC):
    pass


class SatelliteImgSegmentationDM(SegmentationDM):
    def from_s3_data(self, links, batch_size):
        pass


class TiledSatelliteImgSegmentationDM(TiledMixin, SatelliteImgSegmentationDM):
    pass


a = 3
# with sampler
# tiled
# cached
# filtering aka masking

#

# tiled dataset: splits one huge raster image to multiple smaller ones


# class


class Sentinel2SegmentationDM(SegmentationDM):
    def from_s3_data(self, links, batch_size):
        # download only necessary files
        # i.e. necessary band files
        pass


@hydra.main(
    config_path="config/", config_name="infer.yaml", version_base="1.2"
)
def main(cfg):
    lm = LightningModule()
    dm = LightningDataModule()

    # callbacks
    # ckpt_path
    # logger

    ckpt_path = None

    trainer = Trainer()  # get trainer's arguments from cfg

    trainer.predict(lm, dm, ckpt_path)


if __name__ == "__main__":
    main()
