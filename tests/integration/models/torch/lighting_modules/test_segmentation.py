import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
import pytest
from innofw.core.models.torch.lightning_modules.segmentation import SemanticSegmentationLightningModule
from innofw.utils.framework import get_model, get_losses
from innofw.constants import SegDataKeys, Frameworks
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


def predict(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_data = batch[SegDataKeys.image]
            preds = model.predict_proba(input_data)
            predictions.append(preds)
    return torch.cat(predictions, dim=0)


@pytest.fixture
def segmentation_module() -> LightningModule:
    cfg = DictConfig(
        {
            "models": fixt_models.deeplabv3_plus_w_target,
            "trainer": fixt_trainers.trainer_cfg_w_cpu_devices,
            "losses": fixt_losses.jaccard_loss_w_target
        }
    )
    model = get_model(cfg.models, cfg.trainer)
    losses = get_losses(cfg, "image-segmentation", Frameworks.torch)
    optimizer_cfg = DictConfig(fixt_optimizers.adam_optim_w_target)

    module = SemanticSegmentationLightningModule(
        model=model, losses=losses, optimizer_cfg=optimizer_cfg
    )

    return module

def test_training_with_checkpoint(segmentation_module: LightningModule):
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="segmentation_module-{epoch:02d}",
        save_top_k=1,
        monitor="loss",
        mode="min",
        save_weights_only=True,
    )

    trainer = Trainer(max_epochs=1, devices=1, callbacks=[checkpoint_callback])
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4)
    trainer.fit(segmentation_module, train_dataloaders=dataloader)


def test_training_without_checkpoint(segmentation_module: LightningModule):
    trainer = Trainer(max_epochs=1, devices=1)
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4)
    trainer.fit(segmentation_module, train_dataloaders=dataloader)


def test_testing_without_checkpoint(segmentation_module: LightningModule):
    trainer = Trainer(max_epochs=1, devices=1)
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4)
    trainer.test(segmentation_module, dataloaders=dataloader)


def test_predicting_without_checkpoint(segmentation_module: LightningModule):
    trainer = Trainer(max_epochs=1, devices=1)
    predict_dataloader = DataLoader(DummyDataset(10), batch_size=4)
    predictions = predict(segmentation_module, predict_dataloader)