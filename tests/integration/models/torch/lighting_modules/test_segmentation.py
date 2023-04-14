import pytest
import torch
from pytorch_lightning import LightningModule
import pytest
import torch
import time
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from tests.fixtures.config import losses as fixt_losses
from tests.fixtures.config import models as fixt_models
from tests.fixtures.config import optimizers as fixt_optimizers
from tests.fixtures.config import trainers as fixt_trainers
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from innofw.core.models.torch.lightning_modules.segmentation import SemanticSegmentationLightningModule
from innofw.utils.framework import get_model
from innofw.utils.framework import get_losses
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from innofw.constants import SegDataKeys
from innofw.constants import Frameworks
import tempfile
import shutil
import os
from pytorch_lightning.callbacks import ModelCheckpoint


from innofw.core.models.torch.lightning_modules.segmentation import SemanticSegmentationLightningModule


# we should try different losses, different optimizsers and different schedulaers and different models



class DummyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, index):
        x = torch.rand(3, 224, 224)  
        y = torch.randint(0, 2, (1, 224, 224)) 
        return {SegDataKeys.image: x, SegDataKeys.label: y}

    def __len__(self):
        return self.num_samples


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, num_samples: int, batch_size: int = 4):
        super().__init__()
        self.num_samples = num_samples
        self.batch_size = batch_size

    def setup(self, stage= None):
        self.dataset = DummyDataset(self.num_samples)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

@pytest.fixture
def segmentation_module() -> LightningModule:
    cfg = DictConfig(
        {
            "models": fixt_models.deeplabv3_plus_w_target,
            "trainer": fixt_trainers.trainer_cfg_w_cpu_devices,
            "losses" : fixt_losses.jaccard_loss_w_target
        }
    )
    model = get_model(cfg.models, cfg.trainer)
    losses = get_losses(cfg, "image-segmentation", Frameworks.torch) #fixt_losses.jaccard_loss_w_target
    optimizer_cfg = DictConfig(fixt_optimizers.adam_optim_w_target)

    module = SemanticSegmentationLightningModule(
        model=model, losses=losses, optimizer_cfg=optimizer_cfg
    )

    return module


## working ## 

def test_training_with_checkpoint(segmentation_module: LightningModule):
    # Create a directory to save checkpoints
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="segmentation_module-{epoch:02d}",
        save_top_k=1,  # Save only the top 1 checkpoints
        monitor="loss",  # You may need to change this to the appropriate metric you want to monitor
        mode="min",  # Minimize the loss
        save_weights_only=True,
    )

    # Training with checkpoint
    trainer = Trainer(max_epochs=1, devices=1, callbacks=[checkpoint_callback])

    # Create the dummy dataset and dataloader
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4)

    trainer.fit(segmentation_module, train_dataloaders=dataloader)

## working ##

def test_training_without_checkpoint(segmentation_module: LightningModule):
    # training without checkpoint
    trainer = Trainer(max_epochs=1, devices=1)

    # Create the dummy dataset and dataloader
    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4)

    trainer.fit(segmentation_module, train_dataloaders=dataloader)


# working ##
def test_testing_without_checkpoint(segmentation_module: LightningModule):
    # Set up the trainer
    trainer = Trainer(max_epochs=1, devices=1)

    dataset = DummyDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4)

    # Fit the model
    trainer.test(segmentation_module, dataloaders=dataloader)





# def test_testing_without_checkpoint(segmentation_module: LightningModule):
#     checkpoint_path = "checkpoints"
#     os.makedirs(checkpoint_path, exist_ok=True)

#     # Set up the trainer
#     checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, save_top_k=-1)
#     trainer = Trainer(max_epochs=2, devices=1, callbacks=[checkpoint_callback])

#     dataset = DummyDataset(num_samples=100)
#     dataloader = DataLoader(dataset, batch_size=4)

#     # Fit the model
#     trainer.fit(segmentation_module, dataloader)

#     # Load the stored checkpoint
#     loaded_module = segmentation_module.load_from_checkpoint("checkpoints/epoch=1-step=50.ckpt")
    
#     # Test the loaded model
#     trainer.test(loaded_module, dataloaders=dataloader)

#     # Delete the checkpoint folder
#     shutil.rmtree(checkpoint_path)


