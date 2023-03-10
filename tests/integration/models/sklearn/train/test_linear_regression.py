# author: Kazybek Askarbek
# date: 15.07.22
# standard libraries
import pytest
import sklearn.base
from omegaconf import DictConfig

from innofw import InnoModel
from innofw.utils.framework import get_model
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg

# local modules
# basic config

LOGS_FOLDER = "./logs/trainings/runs/default/"

model_cfg_w_target = DictConfig(
    {
        "name": "linear_regression",
        "description": "model by sklearn",
        "_target_": "sklearn.linear_model.LinearRegression",
    }
)

model_cfg_wo_target = model_cfg_w_target.copy()
del model_cfg_wo_target["_target_"]

# _target_ is an empty string
model_cfg_w_empty_target = model_cfg_w_target.copy()
model_cfg_w_empty_target["_target_"] = ""

# _target_ is a None
model_cfg_w_missing_target = model_cfg_w_target.copy()
model_cfg_w_missing_target["_target_"] = None

lasso_model_cfg_w_target = DictConfig(
    {
        "name": "lasso",
        "description": "Linear Model trained with L1 prior as regularizer (aka the Lasso).",
        "_target_": "sklearn.linear_model.Lasso",
    }
)

lasso_model_cfg_w_target_w_argument = lasso_model_cfg_w_target.copy()
lasso_model_cfg_w_target_w_argument["alpha"] = 0.5

ridge_model_cfg_w_target = DictConfig(
    {
        "name": "ridge",
        "description": "Linear Model trained with L2 prior as regularizer (aka the Ridge).",
        "_target_": "sklearn.linear_model.Ridge",
    }
)

ridge_model_cfg_w_target_w_argument = ridge_model_cfg_w_target.copy()
ridge_model_cfg_w_target_w_argument["alpha"] = 0.5


@pytest.mark.parametrize(
    ["cfg"],
    [
        [model_cfg_w_target],
        [model_cfg_wo_target],
        [model_cfg_w_empty_target],
        [model_cfg_w_missing_target],
        [lasso_model_cfg_w_target],
        [lasso_model_cfg_w_target_w_argument],
        [ridge_model_cfg_w_target],
        [ridge_model_cfg_w_target_w_argument],
    ],
)
def test_model_instantiation(cfg):
    model = get_model(cfg, base_trainer_on_cpu_cfg)

    assert isinstance(model, sklearn.base.BaseEstimator)


from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


class MockDatamodule:
    def __init__(self):
        self.setup()

    def setup(self):
        out = load_diabetes()
        X, y = out["data"], out["target"]
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_dataloader(self):
        return {
            "x": self.X_train,
            "y": self.y_train,
        }

    def test_dataloader(self):
        return {
            "x": self.X_test,
            "y": self.y_test,
        }


@pytest.mark.parametrize(
    ["cfg"],
    [
        [model_cfg_w_target],
        [model_cfg_wo_target],
        [model_cfg_w_empty_target],
        [model_cfg_w_missing_target],
        [lasso_model_cfg_w_target],
        [lasso_model_cfg_w_target_w_argument],
        [ridge_model_cfg_w_target],
        [ridge_model_cfg_w_target_w_argument],
    ],
)
def test(cfg):
    model = get_model(cfg, base_trainer_on_cpu_cfg)
    task = "table-regression"
    wrapped_model = InnoModel(model, task=task, log_dir=LOGS_FOLDER)
    dm = MockDatamodule()
    wrapped_model.train(dm)
    wrapped_model.test(dm)
