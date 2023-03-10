import pytest
import sklearn.base
from omegaconf import DictConfig

from innofw.utils.framework import get_model
from innofw.utils.framework import get_obj
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg


def test_sklearn_model_creation():
    cfg = DictConfig(
        {
            "models": {
                "_target_": "sklearn.neighbors.KNeighborsClassifier",
                "name": "knn",
                "description": "something",
            }
        }
    )
    model = get_model(cfg.models, base_trainer_on_cpu_cfg)
    assert isinstance(model, sklearn.base.BaseEstimator)


def test_sklearn_model_creation_name_given():
    cfg = DictConfig(
        {
            "models": {
                "_target_": None,
                "name": "knn-regressor",
                "description": "something",
            }
        }
    )  # "_target_": "???",
    model = get_model(cfg.models, base_trainer_on_cpu_cfg)
    assert isinstance(model, sklearn.base.BaseEstimator)


def test_sklearn_model_n_optimizer_creation():
    cfg = DictConfig(
        {
            "models": {
                "_target_": None,
                "name": "knn-regressor",
                "description": "something",
            },
            "optimizers": {
                "task": ["all"],
                "implementations": {
                    "torch": {
                        "Adam": {
                            "object": {
                                "_target_": "torch.optim.Adam",
                                "lr": 1e-5,
                            }
                        },
                    }
                },
            },
        }
    )
    task = "table-regression"
    framework = "sklearn"
    model = get_model(cfg.models, base_trainer_on_cpu_cfg)
    assert isinstance(model, sklearn.base.BaseEstimator)

    with pytest.raises(AttributeError):
        optim = get_obj(
            cfg, "optimizers", task, framework, params=model.parameters()
        )
