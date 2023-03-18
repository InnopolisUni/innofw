# local modules
import pytest

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules import QsarDataModule
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import qm9_datamodule_cfg_w_target


def test_smoke():
    # create a qsar datamodule
    fw = Frameworks.catboost
    task = "qsar-regression"
    sut: QsarDataModule = get_datamodule(
        qm9_datamodule_cfg_w_target, fw, task=task
    )
    assert sut is not None

    # initialize train and test datasets
    sut.setup()
    assert sut.train_smiles_dataset is not None
    assert sut.test_smiles_dataset is not None


@pytest.mark.parametrize("stage", [Stages.train, Stages.test])
def test_train_datamodule(stage):
    # create a qsar datamodule
    fw = Frameworks.catboost
    task = "qsar-regression"
    sut: QsarDataModule = get_datamodule(
        qm9_datamodule_cfg_w_target, fw, task=task, stage=stage
    )

    # initialize train and test datasets
    sut.setup(stage)

    # get dataloader by stage
    dl = sut.get_stage_dataloader(stage)
    assert dl is not None
    assert dl.get("x") is not None
    assert dl.get("y") is not None
