import os

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
from innofw.core.models.torch.lightning_modules.detection import (
    DetectionLightningModule,
    _evaluate_iou
)

from innofw.utils.framework import get_losses
from innofw.utils.framework import get_model
from innofw.utils.framework import get_datamodule
from tests.fixtures.config import losses as fixt_losses
from tests.fixtures.config import models as fixt_models
from tests.fixtures.config import optimizers as fixt_optimizers
from tests.fixtures.config import trainers as fixt_trainers
from tests.fixtures.config import schedulers as fixt_schedulers
from tests.fixtures.config import datasets as fixt_datasets


class DummyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, index):
        x = torch.rand(3, 224, 224)
        y = torch.randint(0, 2, (1, 224, 224))
        return x, y

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
def dummy_data_module():
    cfg = DictConfig(fixt_datasets.wheat_datamodule_cfg_w_target)
    data_module = get_datamodule(cfg, framework=Frameworks.torch, task="image-detection")
    # data_module = DummyDataModule(num_samples=100, batch_size=4)
    data_module.setup()  # Call the setup method to define the dataset attribute
    return data_module

# @pytest.mark.skip(reason="some bug")
@pytest.fixture(scope="module")
def detection_module() -> LightningModule:
    cfg = DictConfig(
        {
            "models": fixt_models.faster_rcnn_cfg_w_target,
            "trainer": fixt_trainers.trainer_cfg_w_cpu_devices,
            "losses": fixt_losses.focal_loss_w_target,
        }
    )
    model = get_model(cfg.models, cfg.trainer)
    losses = get_losses(cfg, "image-detection", Frameworks.torch)
    optimizer_cfg = DictConfig(fixt_optimizers.adam_optim_w_target)
    scheduler_cfg = DictConfig(fixt_schedulers.linear_w_target)

    module = DetectionLightningModule(
        model=model, losses=losses, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg
    )

    return module   


# @pytest.mark.skip(reason="some bug")
@pytest.fixture(scope="function")
def detection_module_function_scope() -> LightningModule:
    cfg = DictConfig(
        {
            "models": fixt_models.faster_rcnn_cfg_w_target,
            "trainer": fixt_trainers.trainer_cfg_w_cpu_devices,
            "losses": fixt_losses.focal_loss_w_target,
        }
    )
    model = get_model(cfg.models, cfg.trainer)
    losses = get_losses(cfg, "image-detection", Frameworks.torch)
    optimizer_cfg = DictConfig(fixt_optimizers.adam_optim_w_target)
    scheduler_cfg = DictConfig(fixt_schedulers.linear_w_target)

    module = DetectionLightningModule(
        model=model, losses=losses, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg
    )

    return module

# @pytest.mark.skip(reason="some bug")
@pytest.fixture(scope="module")
def fitted_detection_module(
    detection_module: LightningModule,
    trainer_with_temporary_directory,
    dummy_data_module: DummyDataModule,
):
    trainer, _ = trainer_with_temporary_directory
    dataloader = dummy_data_module.train_dataloader()
    trainer.fit(detection_module, train_dataloaders=dataloader)
    detection_module.trainer = (
        trainer  # Add the trainer to the fitted module for future use
    )
    return detection_module

# @pytest.mark.skip(reason="some bug")
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU is found on this machine"
)
def test_training_with_gpu(
    detection_module_function_scope,
    trainer_with_temporary_directory,
    dummy_data_module,
):
    trainer, checkpoint_dir = trainer_with_temporary_directory
    dataloader = dummy_data_module.train_dataloader()
    trainer.fit(detection_module_function_scope, train_dataloaders=dataloader)

# @pytest.mark.skip(reason="some bug")
def test_training_without_checkpoint(
    detection_module_function_scope,
    trainer_with_temporary_directory,
    dummy_data_module,
):
    trainer, checkpoint_dir = trainer_with_temporary_directory
    dataloader = dummy_data_module.train_dataloader()
    trainer.fit(detection_module_function_scope, train_dataloaders=dataloader)

# @pytest.mark.skip(reason="some bug")
def test_training_with_checkpoint(
    fitted_detection_module,
    trainer_with_temporary_directory,
    dummy_data_module,
):
    trainer, checkpoint_dir = trainer_with_temporary_directory
    dataloader = dummy_data_module.train_dataloader()

    # First training phase is already done in the fitted_detection_module fixture
    last_checkpoint_path = (
        fitted_detection_module.trainer.checkpoint_callback.best_model_path
    )

    # Continue training using the fitted_detection_module
    trainer.fit(
        fitted_detection_module,
        ckpt_path=last_checkpoint_path,
        train_dataloaders=dataloader,
    )
@pytest.mark.skip(reason="some bug")
def test_testing_without_checkpoint(
    detection_module_function_scope,
    dummy_data_module,
    trainer_with_temporary_directory,
):
    trainer, checkpoint_dir = trainer_with_temporary_directory
    dataloader = dummy_data_module.train_dataloader()

    # Test the loaded model
    trainer.test(
        detection_module_function_scope,
        dataloaders=dataloader,
    )

@pytest.mark.skip(reason="some bug")
def test_testing_with_checkpoint(
    detection_module_function_scope,
    fitted_detection_module,
    dummy_data_module,
    trainer_with_temporary_directory,
):
    fitted_model_trainer = fitted_detection_module.trainer
    fitted_model_dataloader = dummy_data_module.test_dataloader()

    # Get the checkpoint directory and list all checkpoint files
    checkpoint_dir = fitted_model_trainer.checkpoint_callback.dirpath
    checkpoints = sorted(os.listdir(checkpoint_dir))
    first_checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
    last_checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])

    # Test with the first checkpoint
    first_checkpoint_test_results = fitted_model_trainer.test(
        fitted_detection_module,
        dataloaders=fitted_model_dataloader,
        ckpt_path=first_checkpoint_path,
    )

    # Test with the last checkpoint
    last_checkpoint_test_results = fitted_model_trainer.test(
        fitted_detection_module,
        dataloaders=fitted_model_dataloader,
        ckpt_path=last_checkpoint_path,
    )

    for key in last_checkpoint_test_results[0].keys():
        assert (
            last_checkpoint_test_results[0][key]
            > 0  # first_checkpoint_test_results[0][key]
        )


def test__evaluate_iou():

    target = {"boxes": torch.tensor([[10, 10, 10 ,10]])}
    pred_none = {"boxes": torch.tensor([])}
    iou = _evaluate_iou(target, pred_none)
    assert isinstance(iou, torch.Tensor)

    pred_is = {"boxes": torch.tensor([[15, 15, 10, 10]])}
    iou_ = _evaluate_iou(target, pred_is)
    assert isinstance(iou_, torch.Tensor)