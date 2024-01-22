import os
import shutil

import numpy as np
import pytest
import torch
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from innofw.core.models.sklearn_adapter import SklearnAdapter
from innofw.core.models.torch.architectures.segmentation.unet import UNet
from innofw.core.models.torch.architectures.classification import MobileNetV2
from innofw.core.models.xgboost_adapter import XGBoostAdapter


class MockDatamodule:
    def train_dataloader(self):
        return {
            "x": np.array(
                [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                    [3, 2, 1],
                    [4, 2, 1],
                    [1, 5, 2],
                ]
            ),
            "y": np.array([0, 2, 2, 0, 1, 1]),
        }

    def test_dataloader(self):
        return {
            "x": np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]),
            "y": np.array([1, 2, 2]),
        }

    def setup(self):
        pass


def metric(*args, **kwargs):
    return 1


@pytest.mark.parametrize(
    ["model", "wrapper", "logs"],
    [
        [KNeighborsClassifier, SklearnAdapter, "./logs/test/test1/"],
        [XGBClassifier, XGBoostAdapter, "./logs/test/test2/"],
    ],
)
def test_models(model, wrapper, logs):
    os.makedirs(logs, exist_ok=True)
    l = len(os.listdir(logs))
    model = wrapper(model=model(), metrics=None, log_dir=logs)
    model.metrics = [{"func": metric, "args": {}}]
    model.train(MockDatamodule())
    assert len(os.listdir(logs)) >= l + 1
    shutil.rmtree(logs, ignore_errors=True)


def test_unet():
    x = torch.from_numpy(np.random.random((1, 3, 512, 512)))
    model = UNet()
    assert model.forward(x.float()) is not None

def test_mobilenet():
    x = torch.from_numpy(np.random.random((1, 3, 512, 512)))
    model = MobileNetV2(num_classes=2, pretrained=False)
    assert model.forward(x.float()) is not None
