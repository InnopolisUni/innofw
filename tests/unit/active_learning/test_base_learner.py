import pytest

from innofw.constants import Frameworks
from innofw.core.models.sklearn_adapter import SklearnAdapter
from innofw.core.active_learning.learners import BaseActiveLearner
from innofw.utils.framework import get_datamodule
from innofw.utils.framework import get_model
from tests.fixtures.config.datasets import qm9_datamodule_cfg_w_target
from tests.fixtures.config.models import baselearner_cfg_w_target
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg
from tests.utils import get_test_folder_path

class ConcreteActiveLearner(BaseActiveLearner):
    def eval_model(self, X, y):
        # Implementation of eval_model method
        pass

    def predict_model(self, X):
        # Implementation of predict_model method
        pass

    def obtain_most_uncertain(self, y):
        # Implementation of obtain_most_uncertain method
        pass

# @pytest.mark.skip(reason="some bug")
def test_base_active_learner_creation():
    model = get_model(
        baselearner_cfg_w_target, base_trainer_on_cpu_cfg
    )
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.sklearn, task=task
    )

    sut = ConcreteActiveLearner(
        model=model,
        datamodule=datamodule,
    )

    assert sut is not None

# @pytest.mark.skip(reason="some bug")
def test_base_active_learner_run():
    model = get_model(
        baselearner_cfg_w_target, base_trainer_on_cpu_cfg
    )
    model = SklearnAdapter(model=model, metrics=None, log_dir="./logs/test/test1/")
    
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.sklearn, task=task
    )

    sut = ConcreteActiveLearner(
        model=model,
        datamodule=datamodule,
    )

    sut.run(ckpt_path=str(get_test_folder_path() /'weights/regression_house_prices/lin_reg.pickle'))

    assert sut is not None
