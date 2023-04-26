import os
import shutil

import pytest
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from innofw.constants import SegDataKeys


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


def test_training_with_checkpoint(segmentation_module: LightningModule):
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = Trainer(max_epochs=2, devices=1, default_root_dir=checkpoint_dir)
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4)
    trainer.fit(segmentation_module, train_dataloaders=dataloader)
    trainer.fit(
        segmentation_module,
        ckpt_path="checkpoints/lightning_logs/version_0/checkpoints/epoch=1-step=50.ckpt",
        train_dataloaders=dataloader,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU is found on this machine"
)
def test_training_without_checkpoint(segmentation_module: LightningModule):
    trainer = Trainer(accelerator="gpu", max_epochs=1, devices=1)
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4)
    trainer.fit(segmentation_module, train_dataloaders=dataloader)

    training_metric_values = segmentation_module.training_metric_values

    # Choose a metric to check for improvement, e.g., train_f1
    metric_name = "BinaryF1Score"
    print(training_metric_values)
    # Check if the chosen metric is improving
    for i in range(1, len(training_metric_values)):
        assert (
            training_metric_values[i][metric_name]
            > training_metric_values[i - 1][metric_name]
        )


def test_testing_without_checkpoint(segmentation_module: LightningModule):
    trainer = Trainer(max_epochs=1, accelerator="gpu", devices=1)
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4)
    test_results = trainer.test(segmentation_module, dataloaders=dataloader)

    # Check if test_results is a non-empty list
    assert len(test_results) > 0

    # Check if the first dictionary in test_results contains the expected metrics
    expected_metrics = {
        "test_BinaryF1Score",
        "test_BinaryJaccardIndex",
        "test_BinaryPrecision",
        "test_BinaryRecall",
    }
    for metric in expected_metrics:
        assert metric in test_results[0]


def test_predicting_without_checkpoint(segmentation_module: LightningModule):
    trainer = Trainer(max_epochs=1, devices=1)
    predict_dataloader = DataLoader(DummyDataset(10), batch_size=4)
    # predictions = predict(segmentation_module, predict_dataloader)
    trainer.predict(segmentation_module, dataloaders=predict_dataloader)


def test_testing_with_checkpoint(segmentation_module: LightningModule):
    checkpoint_path = "checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)

    # # Set up the trainer
    # checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, save_top_k=-1)
    trainer = Trainer(
        max_epochs=2, devices=1, default_root_dir=checkpoint_path
    )

    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4)

    # Fit the model
    trainer.fit(segmentation_module, dataloader)

    # Test the loaded model
    trainer.test(
        segmentation_module,
        ckpt_path="checkpoints/lightning_logs/version_0/checkpoints/epoch=1-step=50.ckpt",
        dataloaders=dataloader,
    )

    # Delete the checkpoint folder
    shutil.rmtree(checkpoint_path)


def test_predicting_with_checkpoint(segmentation_module: LightningModule):
    checkpoint_path = "checkpoints"

    # Instantiate the DummyDataModule
    data_module = DummyDataModule(num_samples=100, batch_size=4)

    # Instantiate the Trainer with desired configurations
    trainer = Trainer(
        max_epochs=2, devices=1, default_root_dir=checkpoint_path
    )

    # Fit the model (train and validate)
    trainer.fit(segmentation_module, data_module)

    # Create a DataLoader for the prediction data
    predict_dataloader = DataLoader(DummyDataset(10), batch_size=4)

    trainer.predict(
        segmentation_module,
        ckpt_path="checkpoints/lightning_logs/version_0/checkpoints/epoch=1-step=50.ckpt",
        dataloaders=predict_dataloader,
    )
