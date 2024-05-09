# author: Kazybek Askarbek
# date: 12.07.22
# standard libraries
import os
import pytest
import shutil
from itertools import product
from omegaconf import DictConfig

from innofw import InnoModel
from innofw.constants import Frameworks
from ultralytics import YOLO
from innofw.utils.framework import get_datamodule
from innofw.utils.framework import get_model
from innofw.core.integrations.ultralytics.optimizers import UltralyticsOptimizerBaseAdapter
from innofw.core.integrations.ultralytics.losses import UltralyticsLossesBaseAdapter
from innofw.core.integrations.ultralytics.yolov5_adapter import YOLOV5Adapter, YOLOv5Model, YOLOV5_VALID_ARCHS
from tests.fixtures.config.datasets import lep_datamodule_cfg_w_target, stroke_segmentation_datamodule_cfg_w_target
from tests.fixtures.config.models import yolov5_cfg_w_target
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg, base_trainer_on_gpu_cfg
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

@pytest.mark.parametrize(
    ["cfg"],
    [
        [stroke_segmentation_datamodule_cfg_w_target],
    ],
)
def test_wrong_datamodule_instantiation(cfg):
    task = "image-detection"
    framework = Frameworks.ultralytics
    with pytest.raises(ValueError):
        datamodule = get_datamodule(cfg, framework, task=task)


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


@pytest.mark.parametrize(
    "optim_cfg",
    [
        None,
        DictConfig({'_target_': 'torch.optim.SGD', 'lr0': 3e-4}),
        DictConfig({'_target_': 'torch.optim.Adam', 'lrf': 1e-5}),
        DictConfig({'_target_': 'torch.optim.AdamW', 'lrf': 1e-5, 'lr0': 3e-4}),
    ],
)
def test_different_optimizer_selection(optim_cfg):
    optim = UltralyticsOptimizerBaseAdapter().adapt(optim_cfg)
    assert len(UltralyticsOptimizerBaseAdapter().from_cfg(optim_cfg))
    assert len(UltralyticsOptimizerBaseAdapter().from_obj(None))


def test_losses():
    loss_adapter = UltralyticsLossesBaseAdapter()
    assert len(loss_adapter.from_cfg(None)) == 2
    assert len(loss_adapter.from_obj(None)) == 2


@pytest.mark.parametrize(
    ["model", "trainer_cfg", "weights_freq"],
    [
        list(tup) for tup in product([YOLOv5Model(arch) for arch in YOLOV5_VALID_ARCHS],
                                     [base_trainer_on_cpu_cfg, base_trainer_on_gpu_cfg],
                                     [None, 1])
    ]
)
def test_adapter_creation(model,
                          trainer_cfg,
                          weights_freq):
    adapter = YOLOV5Adapter(model, './tmp', trainer_cfg, weights_freq=weights_freq)

    adapter.update_checkpoints_path()

    assert adapter._yolov5_train is not None
    assert adapter._yolov5_val is not None
    assert adapter._yolov5_predict is not None

def test_yolo_train():
    os.makedirs('./tmp', exist_ok=True)
    datamodule = get_datamodule(lep_datamodule_cfg_w_target, Frameworks.ultralytics, task="image-detection")

    adapter = YOLOV5Adapter(YOLOv5Model("yolov5s"), './tmp', base_trainer_on_cpu_cfg)
    adapter.train(datamodule, ckpt_path=None)

    for i in range(3):
        try:
            shutil.rmtree('./tmp')
            shutil.rmtree('./something')
            break
        except:
            pass

# def test_model_training + with metrics
# def test_model_logging
# def test_different_scheduler_selection
# def test_wrong_scheduler_selection
# def test_how_weight_initialization work on yolov5 model
# def test_various_checkpointing options
# def test_metrics_calculation
# def test_different_batch_size
# def test_epochs_specified
# def test_callbacks
