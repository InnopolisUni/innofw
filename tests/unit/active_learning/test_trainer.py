import os

from innofw import InnoModel
from innofw.constants import Frameworks
from innofw.core.active_learning import ActiveLearnTrainer
from innofw.utils.framework import get_datamodule
from innofw.utils.framework import get_model
from tests.fixtures.config.datasets import qm9_datamodule_cfg_w_target
from tests.fixtures.config.models import catboost_with_uncertainty_cfg_w_target
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg


def test_active_learn_trainer_creation():
    model = InnoModel(
        get_model(
            catboost_with_uncertainty_cfg_w_target, base_trainer_on_cpu_cfg
        ),
        log_dir=os.devnull,
    )
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.catboost, task=task
    )

    sut = ActiveLearnTrainer(model=model, datamodule=datamodule)

    assert sut is not None
