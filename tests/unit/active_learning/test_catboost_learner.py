from innofw.core.active_learning.learners import CatBoostActiveLearner
from innofw.constants import Frameworks
from innofw.utils.framework import get_datamodule, get_model
from tests.fixtures.config.datasets import qm9_datamodule_cfg_w_target
from tests.fixtures.config.models import catboost_with_uncertainty_cfg_w_target
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg


def test_catboost_active_learner_creation():
    model = get_model(catboost_with_uncertainty_cfg_w_target, base_trainer_on_cpu_cfg)
    task = "qsar-regression"
    datamodule = get_datamodule(qm9_datamodule_cfg_w_target, Frameworks.catboost, task=task)

    sut = CatBoostActiveLearner(
        model=model,
        datamodule=datamodule,
    )

    assert sut is not None
