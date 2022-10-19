import os

from innofw.core.active_learning import ActiveLearnTrainer
from innofw.constants import Frameworks
from innofw.utils.framework import get_datamodule, get_model
from innofw import InnoModel
from tests.fixtures.config.datasets import qm9_datamodule_cfg_w_target
from tests.fixtures.config.models import catboost_with_uncertainty_cfg_w_target
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg


def test_active_learn_trainer_creation():
    model = InnoModel(
        get_model(catboost_with_uncertainty_cfg_w_target, base_trainer_on_cpu_cfg),
        log_dir=os.devnull,
    )
    datamodule = get_datamodule(qm9_datamodule_cfg_w_target, Frameworks.catboost)

    sut = ActiveLearnTrainer(model=model, datamodule=datamodule)

    assert sut is not None
