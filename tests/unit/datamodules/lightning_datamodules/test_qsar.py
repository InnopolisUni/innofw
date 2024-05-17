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



def test_train_datamodule():
    # create a qsar datamodule
    fw = Frameworks.torch
    task = "text-vae"
    dm: QsarSelfiesDataModule = get_datamodule(
        qsar_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup()

    # get dataloader by stage
    dltr = dm.get_stage_dataloader(Stages.train)
    dlte = dm.get_stage_dataloader(Stages.test)

    dlval = dm.val_dataloader()

    dm.setup_infer()
    dlinf = dm.get_stage_dataloader(Stages.predict)

    assert dltr is not None
    assert dlte is not None
    assert dlinf is not None
    assert dlval is not None

