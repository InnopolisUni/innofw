import pytest
import sklearn
import torch
import xgboost
import catboost
import ultralytics
from omegaconf import DictConfig

from innofw import InnoModel
from innofw.constants import Frameworks
from innofw.utils.framework import get_datamodule
from innofw.utils.framework import get_model
from innofw.utils.framework import map_model_to_framework, is_suitable_for_framework
from innofw.core.integrations.base_integration_models import BaseIntegrationModel
from tests.fixtures.config.datasets import house_prices_datamodule_cfg_w_target
from tests.fixtures.config.datasets import lep_datamodule_cfg_w_target
from tests.fixtures.config.models import linear_regression_cfg_w_target
from tests.fixtures.config.models import xgbregressor_cfg_w_target
from tests.fixtures.config.models import yolov5_cfg_w_target



@pytest.mark.parametrize(
    ["model", "gt_framework"],
    [
        [sklearn.linear_model.LinearRegression(), Frameworks.sklearn],
        [xgboost.XGBRegressor(), Frameworks.xgboost],
        [torch.nn.Linear(10, 10), Frameworks.torch],
        [BaseIntegrationModel(), BaseIntegrationModel().framework],
        [catboost.CatBoostRegressor(), Frameworks.catboost],
        [ultralytics.YOLO(), Frameworks.ultralytics]
    ]

)
def test_map_model_to_framework(model, gt_framework):
    assert map_model_to_framework(model) == gt_framework

def test_unsupported_framework():
    with pytest.raises(NotImplementedError):
        map_model_to_framework(None)

def test_is_suitable_for_framework():
    assert is_suitable_for_framework(None, "adapter")
    # assert is_suitable_for_framework(linear_model, Frameworks.sklearn)
    # assert not is_suitable_for_framework(linear_model, Frameworks.torch)

def test_is_suitable_for_task():
    pass

def test_get_augmentations():
    pass