from innofw.constants import Frameworks
from innofw.core.models.sklearn_adapter import SklearnAdapter
from innofw.core.active_learning.learners import BaseActiveLearner
from innofw.utils.framework import get_datamodule
from innofw.utils.framework import get_model
from tests.fixtures.config.datasets import qm9_datamodule_cfg_w_target
from tests.fixtures.config.models import baselearner_cfg_w_target
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg


def test_base_active_learner_creation():
    model = get_model(
        baselearner_cfg_w_target, base_trainer_on_cpu_cfg
    )
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.sklearn, task=task
    )

    sut = BaseActiveLearner(
        model=model,
        datamodule=datamodule,
    )

    assert sut is not None


def test_base_active_learner_run():
    model = get_model(
        baselearner_cfg_w_target, base_trainer_on_cpu_cfg
    )
    model = SklearnAdapter(model=model, metrics=None, log_dir="./logs/test/test1/")
    
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.sklearn, task=task
    )

    sut = BaseActiveLearner(
        model=model,
        datamodule=datamodule,
    )

    sut.run(ckpt_path='/workspace/innofw/tests/weights/regression_house_prices/lin_reg.pickle')

    assert sut is not None
