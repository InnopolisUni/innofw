import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from innofw.constants import Frameworks
from innofw.constants import SegDataKeys, SegOutKeys
from innofw.core.models.torch.lightning_modules.segmentation import (
    MulticlassSemanticSegmentationLightningModule
)
from innofw.utils.framework import get_losses
from innofw.utils.framework import get_model
from tests.fixtures.config import losses as fixt_losses
from tests.fixtures.config import models as fixt_models
from tests.fixtures.config import optimizers as fixt_optimizers
from tests.fixtures.config import trainers as fixt_trainers


class MultiSegDummyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, index):
        x = torch.rand(3, 224, 224)
        y = torch.randint(0, 4, (1, 224, 224))
        return {SegDataKeys.image: x, SegDataKeys.label: y}

    def __len__(self):
        return self.num_samples


class MultiSegDummyDataModule(LightningDataModule):
    def __init__(self, num_samples: int, batch_size: int = 4):
        super().__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = MultiSegDummyDataset(self.num_samples)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)


def test_multiclasssegmentation_module() -> LightningModule:
    cfg = DictConfig(
        {
            "models": fixt_models.deeplabv3_plus_w_target_multiclass,
            "trainer": fixt_trainers.trainer_cfg_w_cpu_devices,
            "losses": fixt_losses.multiclass_jaccard_loss_w_target,
        }
    )
    model = get_model(cfg.models, cfg.trainer)
    losses = get_losses(cfg, "multiclass-image-segmentation", Frameworks.torch)
    optimizer_cfg = DictConfig(fixt_optimizers.adam_optim_w_target)

    module = MulticlassSemanticSegmentationLightningModule(
        model=model, losses=losses, optimizer_cfg=optimizer_cfg
    )

    assert module is not None

    datamodule = MultiSegDummyDataModule(num_samples=8)
    datamodule.setup()

    for stage in ["train", "val"]:
        output = module.stage_step(stage, next(iter(datamodule.train_dataloader())),
                                   do_logging=True)
        assert output[SegOutKeys.predictions].shape == (4, 4, 224, 224)
