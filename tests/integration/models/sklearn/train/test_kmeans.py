# author: Kazybek Askarbek
# date: 15.07.22
# standard libraries
import sklearn.base
from omegaconf import DictConfig
import pytest

# local modules
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg
from innofw.utils.framework import get_datamodule, get_model

# basic config
from innofw import InnoModel

LOGS_FOLDER = (
    "./logs/trainings/runs/default/"
    # todo: add test with varying logger folders  # todo: better to log into tmp folders
)

model_cfg_w_target = DictConfig(
    {
        "name": "kmeans",
        "description": "description",
        "_target_": "sklearn.cluster.KMeans",
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


@pytest.mark.parametrize(
    ["cfg"],
    [
        [model_cfg_w_target],
        [model_cfg_wo_target],
        [model_cfg_w_empty_target],
        [model_cfg_w_missing_target],
    ],
)
def test_model_instantiation(cfg):
    model = get_model(cfg, base_trainer_on_cpu_cfg)

    assert isinstance(model, sklearn.base.BaseEstimator)


from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


class BaseMockDM:
    def __init__(self):
        self.setup()

    def setup(self):
        raise NotImplementedError

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


class MockDatamoduleWithTarget(BaseMockDM):
    def setup(self):
        X, y = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )


class MockDatamoduleWithOutTarget(BaseMockDM):
    def setup(self):
        X, y = make_blobs(n_samples=100, centers=3, n_features=3, random_state=42)
        self.X_train, self.X_test = train_test_split(X, test_size=0.2, random_state=42)

        self.y_train, self.y_test = None, None


@pytest.mark.parametrize(
    ["cfg"],
    [
        [model_cfg_w_target],
        [model_cfg_wo_target],
        [model_cfg_w_empty_target],
        [model_cfg_w_missing_target],
    ],
)
def test_w_y(cfg):
    model = get_model(cfg, base_trainer_on_cpu_cfg)
    task = "table-regression"
    wrapped_model = InnoModel(model=model, task=task, log_dir=LOGS_FOLDER)
    dm = MockDatamoduleWithTarget()
    wrapped_model.train(dm)
    wrapped_model.test(dm)


@pytest.mark.parametrize(
    ["cfg"],
    [
        [model_cfg_w_target],
        [model_cfg_wo_target],
        [model_cfg_w_empty_target],
        [model_cfg_w_missing_target],
    ],
)
def test_wo_y(cfg):
    model = get_model(cfg, base_trainer_on_cpu_cfg)
    task = "table-regression"
    wrapped_model = InnoModel(model=model, task=task, log_dir=LOGS_FOLDER)
    dm = MockDatamoduleWithOutTarget()
    wrapped_model.train(dm)
    wrapped_model.test(dm)
