from innofw.constants import Frameworks
from innofw.core.active_learning.datamodule import ActiveDataModule
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import qm9_datamodule_cfg_w_target


def test_active_datamodule_creation():
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.catboost, task=task
    )
    sut = ActiveDataModule(datamodule=datamodule)

    assert sut is not None
