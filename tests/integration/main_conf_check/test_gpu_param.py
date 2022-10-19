# standard libraries

# third party libraries
import pytest
import torch

# local modules


# model + dataset cfg options:
# ===============================
from omegaconf import DictConfig

from innofw.utils.framework import (
    get_datamodule,
    get_model,
    map_model_to_framework,
)
from innofw import InnoModel
from tests.fixtures.config.models import (
    yolov5_cfg_w_target,
    linear_regression_cfg_w_target,
    xgbregressor_cfg_w_target,
)
from tests.fixtures.config.trainers import (
    base_trainer_on_cpu_cfg,
    trainer_cfg_w_gpu_devices_1,
    trainer_cfg_w_gpu_single_gpu0,
)
from tests.fixtures.config.datasets import (
    lep_datamodule_cfg_w_target,
    house_prices_datamodule_cfg_w_target,
)

# NOTE:
# trainer_cfg options:
# ===============================
# If the devices stage is not defined, it will assume devices to be "auto"
# and fetch the auto_device_count from the accelerator.


# ================= TEST CASES =================
arable_segmentation_datamodule_cfg_w_target = None

from innofw.core.integrations.ultralytics.trainer import get_device


@pytest.mark.parametrize(["trainer_cfg"], [[base_trainer_on_cpu_cfg]])
def test_get_device(trainer_cfg):
    device = get_device(trainer_cfg)
    assert device == "cpu"


@pytest.mark.parametrize(
    ["model_cfg", "dm_cfg", "trainer_cfg", "task"],
    [
        [
            yolov5_cfg_w_target,
            lep_datamodule_cfg_w_target,
            base_trainer_on_cpu_cfg,
            "image-detection",
        ],
        # [
        #     linear_regression_cfg_w_target,
        #     house_prices_datamodule_cfg_w_target,
        #     base_trainer_on_cpu_cfg,
        #     "table-regression",
        # ],
        # [
        #     xgbregressor_cfg_w_target,
        #     house_prices_datamodule_cfg_w_target,
        #     base_trainer_on_cpu_cfg,
        #     "table-regression",
        # ],
        # [unet_cfg_w_target, arable_segmentation_datamodule_cfg_w_target, base_trainer_on_cpu_cfg, "image-segmentation"],
    ],
)
@pytest.mark.skip
def test_on_cpu(model_cfg, dm_cfg, trainer_cfg, task):
    from innofw.pipeline import run_pipeline

    cfg = DictConfig(
        {
            "models": model_cfg,
            "datasets": dm_cfg,
            "trainer": trainer_cfg,
            "task": task,
            "batch_size": 3,
            "epochs": 1,
            "accelerator": "cpu",
            "project": "gpu_testing",
            "experiment": "something",
        }
    )
    run_pipeline(cfg, train=True)
    # # create a model
    # model = get_model(model_cfg, trainer_cfg)
    # # create a dataset
    # framework = map_model_to_framework(model)
    # dm = get_datamodule(dm_cfg, framework=framework)
    # # wrap a model
    # wrp_model = Wrapper.wrap(
    #     model, trainer_cfg=trainer_cfg, task=task, log_dir="./logs/test"
    # )
    # # start training
    # wrp_model.train(dm)


# run these tests only if gpu is available on the machine
@pytest.mark.parametrize(
    ["model_cfg", "dm_cfg", "trainer_cfg", "task"],
    [
        [
            linear_regression_cfg_w_target,
            house_prices_datamodule_cfg_w_target,
            base_trainer_on_cpu_cfg,
            "table-regression",
        ],
        [
            linear_regression_cfg_w_target,
            house_prices_datamodule_cfg_w_target,
            trainer_cfg_w_gpu_devices_1,
            "table-regression",
        ],
        [
            linear_regression_cfg_w_target,
            house_prices_datamodule_cfg_w_target,
            trainer_cfg_w_gpu_single_gpu0,
            "table-regression",
        ],
        [
            xgbregressor_cfg_w_target,
            house_prices_datamodule_cfg_w_target,
            base_trainer_on_cpu_cfg,
            "table-regression",
        ],
        [
            xgbregressor_cfg_w_target,
            house_prices_datamodule_cfg_w_target,
            trainer_cfg_w_gpu_devices_1,
            "table-regression",
        ],
        [
            xgbregressor_cfg_w_target,
            house_prices_datamodule_cfg_w_target,
            trainer_cfg_w_gpu_single_gpu0,
            "table-regression",
        ],
        # [unet_cfg_w_target, arable_segmentation_datamodule_cfg_w_target, base_trainer_on_cpu_cfg, "image-segmentation"],
    ],
)
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU is found on this machine"
)
def test_on_gpu(model_cfg, dm_cfg, trainer_cfg, task):
    # def test_on_gpu():
    # create a model
    model = get_model(model_cfg, trainer_cfg)
    # create a dataset
    framework = map_model_to_framework(model)
    dm = get_datamodule(dm_cfg, framework=framework)
    # wrap a model
    wrp_model = InnoModel(
        model, trainer_cfg=trainer_cfg, task=task, log_dir="./logs/test"
    )
    # start training
    wrp_model.train(dm)

    assert True


# # [
# #     yolov5_cfg_w_target,
# #     lep_datamodule_cfg_w_target,
# #     base_trainer_on_gpu_cfg,
# #     "image-detection",
# # ],
# [
#     yolov5_cfg_w_target,
#     lep_datamodule_cfg_w_target,
#     trainer_cfg_w_gpu_devices_1,
#     "image-detection",
# ],
#
#
# # [
# #     yolov5_cfg_w_target,
# #     lep_datamodule_cfg_w_target,
# #     trainer_cfg_w_gpu_single_gpu0,
# #     "image-detection",
# # ],

# # test on gpu 1
# @pytest.mark.parametrize(
#     ["trainer_cfg"],
#     [
#         [trainer_cfg_w_gpu_single_gpu1],
#     ],
# )
# @pytest.mark.skipif(torch.cuda.device_count() <= 1, reason="Not enough GPU devices on this machine")
# def test_on_gpu(trainer_cfg):
#     pass
#
# # todo: test model inference
# # todo: test model testing

# todo: test yolov5 using two gpu devices # for now it freezes
