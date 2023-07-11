# author: Kazybek Askarbek
# date: 12.07.22
# standard libraries
import pytest

from innofw import InnoModel
from innofw.constants import Frameworks
from ultralytics import YOLO
from innofw.utils.framework import get_datamodule
from innofw.utils.framework import get_model
from tests.fixtures.config.datasets import lep_datamodule_cfg_w_target
from tests.fixtures.config.models import yolov5_cfg_w_target
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg
from tests.utils import get_test_folder_path

# local modules
# basic config

# no _target_ key
model_cfg_wo_target = yolov5_cfg_w_target.copy()
del model_cfg_wo_target["_target_"]

# _target_ is an empty string
model_cfg_w_empty_target = yolov5_cfg_w_target.copy()
model_cfg_w_empty_target["_target_"] = ""

# _target_ is a None
model_cfg_w_missing_target = yolov5_cfg_w_target.copy()
model_cfg_w_missing_target["_target_"] = None


@pytest.mark.parametrize(
    ["cfg"],
    [
        [yolov5_cfg_w_target],
        [model_cfg_wo_target],
        [model_cfg_w_empty_target],
        [model_cfg_w_missing_target],
    ],
)
def test_model_instantiation(cfg):
    model = get_model(cfg, base_trainer_on_cpu_cfg)

    assert isinstance(model, YOLO)


# def test_model_instantiation_wrong_data
@pytest.mark.parametrize(
    ["cfg"],
    [
        [lep_datamodule_cfg_w_target],
    ],
)
def test_datamodule_instantiation(cfg):
    task = "image-detection"
    framework = Frameworks.ultralytics
    datamodule = get_datamodule(cfg, framework, task=task)


@pytest.mark.parametrize(
    ["model_cfg", "dm_cfg"],
    [
        [yolov5_cfg_w_target, lep_datamodule_cfg_w_target],
        # [model_cfg_wo_target, datamodule_cfg_w_target],
        # [model_cfg_w_empty_target, datamodule_cfg_w_target],
        # [model_cfg_w_missing_target, datamodule_cfg_w_target],
    ],
)
def test_model_predicting(model_cfg, dm_cfg):
    ckpt_path = str(get_test_folder_path() / "weights/detection_lep/best.pt")
    model = get_model(model_cfg, base_trainer_on_cpu_cfg)
    task = "image-detection"
    framework = Frameworks.ultralytics
    datamodule = get_datamodule(dm_cfg, framework, task=task)

    wrapped_model = InnoModel(
        model,
        task=task,
        trainer_cfg=base_trainer_on_cpu_cfg,
        log_dir="./logs/test/logs",
    )
    wrapped_model.predict(datamodule, ckpt_path=ckpt_path)


# def test_model_training + with metrics
# def test_model_logging
# def test_different_optimizer_selection
# def test_wrong_optimizer_selection
# def test_different_scheduler_selection
# def test_wrong_scheduler_selection
# def test_how_weight_initialization work on yolov5 model
# def test_augmentations_selection on yolov5 model
# def test_various_checkpointing options
# def test_metrics_calculation
# def test_different_batch_size
# def test_epochs_specified
# def test_losses
# def test_callbacks
