import os
import tempfile

import pytest
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from innofw.constants import Frameworks
from innofw.constants import SegDataKeys
from innofw.core.models.torch.lightning_modules.segmentation import (
    SemanticSegmentationLightningModule,
)
from innofw.utils.framework import get_losses
from innofw.utils.framework import get_model
from tests.fixtures.config import losses as fixt_losses
from tests.fixtures.config import models as fixt_models
from tests.fixtures.config import optimizers as fixt_optimizers
from tests.fixtures.config import trainers as fixt_trainers


class DummyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, index):
        x = torch.rand(3, 224, 224)
        y = torch.randint(0, 2, (1, 224, 224))
        return {SegDataKeys.image: x, SegDataKeys.label: y}

    def __len__(self):
        return self.num_samples


class DummyDataModule(LightningDataModule):
    def __init__(self, num_samples: int, batch_size: int = 4):
        super().__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = DummyDataset(self.num_samples)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)


@pytest.fixture(scope="module")
def segmentation_module() -> LightningModule:
    cfg = DictConfig(
        {
            "models": fixt_models.deeplabv3_plus_w_target,
            "trainer": fixt_trainers.trainer_cfg_w_cpu_devices,
            "losses": fixt_losses.jaccard_loss_w_target,
        }
    )
    model = get_model(cfg.models, cfg.trainer)
    losses = get_losses(cfg, "image-segmentation", Frameworks.torch)
    optimizer_cfg = DictConfig(fixt_optimizers.adam_optim_w_target)

    module = SemanticSegmentationLightningModule(
        model=model, losses=losses, optimizer_cfg=optimizer_cfg
    )

    return module


@pytest.fixture(scope="module")
def trainer_with_temporary_directory():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tmp_dir, "checkpoints"),
            every_n_epochs=1,
            save_top_k=-1,
        )

        trainer = Trainer(
            max_epochs=5,
            devices=1,
            accelerator="gpu" if torch.cuda.is_available() else None,
            default_root_dir=tmp_dir,
            callbacks=[checkpoint_callback],
        )
        yield trainer, tmp_dir


@pytest.fixture(scope="module")
def dummy_data_module():
    data_module = DummyDataModule(num_samples=100, batch_size=4)
    data_module.setup()  # Call the setup method to define the dataset attribute
    return data_module


@pytest.fixture(scope="function")
def segmentation_module_function_scope() -> LightningModule:
    cfg = DictConfig(
        {
            "models": fixt_models.deeplabv3_plus_w_target,
            "trainer": fixt_trainers.trainer_cfg_w_cpu_devices,
            "losses": fixt_losses.jaccard_loss_w_target,
        }
    )
    model = get_model(cfg.models, cfg.trainer)
    losses = get_losses(cfg, "image-segmentation", Frameworks.torch)
    optimizer_cfg = DictConfig(fixt_optimizers.adam_optim_w_target)

    module = SemanticSegmentationLightningModule(
        model=model, losses=losses, optimizer_cfg=optimizer_cfg
    )

    return module


@pytest.fixture(scope="module")
def fitted_segmentation_module(
    segmentation_module: LightningModule,
    trainer_with_temporary_directory,
    dummy_data_module: DummyDataModule,
):
    trainer, _ = trainer_with_temporary_directory
    dataloader = dummy_data_module.train_dataloader()
    trainer.fit(segmentation_module, train_dataloaders=dataloader)
    segmentation_module.trainer = (
        trainer  # Add the trainer to the fitted module for future use
    )
    return segmentation_module
