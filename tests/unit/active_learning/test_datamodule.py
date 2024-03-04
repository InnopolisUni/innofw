from innofw.constants import Frameworks
from innofw.core.active_learning.datamodule import ActiveDataModule, get_active_datamodule
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import qm9_datamodule_cfg_w_target


def test_active_datamodule_creation():
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.catboost, task=task
    )
    sut = ActiveDataModule(datamodule=datamodule)
    assert sut is not None
    sut.setup()
    assert len(sut.train_dataloader()) == 2


def test_get_active_datamodule():
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.catboost, task=task
    )
    sut = get_active_datamodule
    assert sut is not None
