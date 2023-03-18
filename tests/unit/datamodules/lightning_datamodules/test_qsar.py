import pytest

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.lightning_datamodules.qsar_dm import (
    QsarSelfiesDataModule,
)
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import qsar_datamodule_cfg_w_target

# local modules


def test_smoke():
    # create a qsar datamodule
    fw = Frameworks.torch
    task = "text-vae"
    dm: QsarSelfiesDataModule = get_datamodule(
        qsar_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup()

    for item in [
        dm.work_mode,
        dm.val_size,
        dm.collator,
    ]:
        assert item is not None


@pytest.mark.parametrize("stage", [Stages.train, Stages.test])
def test_train_datamodule(stage):
    # create a qsar datamodule
    fw = Frameworks.torch
    task = "text-vae"
    dm: QsarSelfiesDataModule = get_datamodule(
        qsar_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup(stage)

    # get dataloader by stage
    dl = dm.get_stage_dataloader(stage)
    assert dl is not None
