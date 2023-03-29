# other
import hydra.utils
import pytest
import torch.nn as nn
from omegaconf import DictConfig

from innofw import InnoModel
from innofw.constants import Frameworks
from innofw.utils.framework import get_datamodule
from innofw.utils.framework import get_model
from innofw.utils.framework import get_obj
from innofw.utils.framework import get_optimizer
from tests.fixtures.config import losses as fixt_losses
from tests.fixtures.config import models as fixt_models
from tests.fixtures.config import optimizers as fixt_optimizers
from tests.fixtures.config import trainers as fixt_trainers
from tests.fixtures.config.augmentations import resize_augmentation_albu
from tests.fixtures.config.augmentations import resize_augmentation_torchvision
from tests.fixtures.config.datasets import (
    faces_datamodule_cfg_w_target,
)
from tests.fixtures.config.models import (
    resnet_cfg_w_target,
)

# local


@pytest.mark.parametrize(
    ["target"],
    [
        ["segmentation_models_pytorch.DeepLabV3Plus"],
        [
            "innofw.core.models.torch.architectures.detection.faster_rcnn.FasterRcnnModel"
        ],
        [
            "innofw.core.models.torch.architectures.classification.resnet.Resnet18"
        ],
        ["innofw.core.models.torch.architectures.segmentation.SegFormer"],
    ],
)
def test_model_creation(target):
    cfg = DictConfig(
        {
            "models": {
                "_target_": target,
                "name": "test",
                "description": "something",
            }
        }
    )
    model = get_model(cfg.models, fixt_trainers.base_trainer_on_cpu_cfg)
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize(
    ["name"],
    [["unet"], ["FasterRcnnModel"], ["resnet18"], ["SegFormer"]],
)
def test_model_creation_name_given(name):
    cfg = DictConfig(
        {
            "models": {
                "_target_": None,
                "name": name,
                "description": "something",
            }
        }
    )
    model = get_model(cfg.models, fixt_trainers.base_trainer_on_cpu_cfg)
    assert isinstance(model, nn.Module)


def test_model_creation_with_arguments():
    cfg = DictConfig(
        {
            "models": {
                "name": "deeplabv3plus",
                "description": "something",
                "_target_": "segmentation_models_pytorch.DeepLabV3Plus",
                "encoder_name": "dpn98",
                "encoder_weights": None,
                "classes": 1,
                "activation": "sigmoid",
                "in_channels": 4,
            }
        }
    )
    model = get_model(cfg.models, fixt_trainers.base_trainer_on_cpu_cfg)
    assert isinstance(model, nn.Module)


def test_model_n_optimizer_creation():
    cfg = DictConfig(
        {
            "models": fixt_models.deeplabv3_plus_w_target,
            "optimizers": fixt_optimizers.adam_optim_w_target,
        }
    )
    task = "image-segmentation"
    framework = Frameworks.torch
    model = get_model(cfg.models, fixt_trainers.base_trainer_on_cpu_cfg)
    optim_cfg = get_optimizer(cfg, "optimizers", task, framework)
    optim = hydra.utils.instantiate(optim_cfg, params=model.parameters())


def test_torch_wrapper_creation():
    cfg = DictConfig(
        {
            "models": fixt_models.deeplabv3_plus_w_target,
            "optimizers": fixt_optimizers.adam_optim_w_target,
            "losses": fixt_losses.jaccard_loss_w_target,
        }
    )
    task = "image-segmentation"
    framework = Frameworks.torch
    model = get_model(cfg.models, fixt_trainers.base_trainer_on_cpu_cfg)
    losses = get_obj(cfg, "losses", task, framework)
    wrapped_model = InnoModel(
        model, task=task, losses=losses, log_dir="./logs/test/"
    )

    assert wrapped_model is not None


@pytest.mark.parametrize(
    ["model_cfg", "dm_cfg", "task", "aug"],
    [
        # [yolov5_cfg_w_target, lep_datamodule_cfg_w_target, "image-detection", None],
        # [faster_rcnn_cfg_w_target, wheat_datamodule_cfg_w_target, "image-detection", None],
        [
            resnet_cfg_w_target,
            faces_datamodule_cfg_w_target,
            "image-classification",
            resize_augmentation_torchvision,
        ],
        [
            resnet_cfg_w_target,
            faces_datamodule_cfg_w_target,
            "image-classification",
            resize_augmentation_albu,
        ]
        # [model_cfg_wo_target, datamodule_cfg_w_target],
        # [model_cfg_w_empty_target, datamodule_cfg_w_target],
        # [model_cfg_w_missing_target, datamodule_cfg_w_target],
    ],
)
def test_model_training(model_cfg, dm_cfg, task, aug):
    model = get_model(model_cfg, fixt_trainers.base_trainer_on_cpu_cfg)
    framework = Frameworks.torch
    augmentations = (
        None if not aug else get_obj(aug, "augmentations", task, framework)
    )
    datamodule = get_datamodule(
        dm_cfg,
        framework,
        task=task,
        augmentations={
            "train": augmentations,
            "test": augmentations,
            "val": augmentations,
        },
    )

    wrapped_model = InnoModel(
        model,
        task=task,
        trainer_cfg=fixt_trainers.base_trainer_on_cpu_cfg,
        log_dir="./logs/test/logs",
    )
    wrapped_model.train(datamodule)
