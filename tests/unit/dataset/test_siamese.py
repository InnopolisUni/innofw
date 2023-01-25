import pytest

# local modules
from innofw.constants import Frameworks, Stages
from innofw.utils.framework import get_datamodule
from innofw.core.datamodules.lightning_datamodules.siamese_dm import SiameseDataModule
from tests.fixtures.config.datasets import faces_siamese_datamodule_cfg_w_target


def test_smoke():
    # create a qsar datamodule
    fw = Frameworks.torch
    task = "one-shot-learning"
    dm: SiameseDataModule = get_datamodule(
        faces_siamese_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup(Stages.train)

    for item in [
        dm.val_size,
        dm.train_dataset,
        dm.val_dataset,
    ]:
        assert item is not None

    dm.setup(Stages.predict)

    assert dm.predict_dataset is not None


@pytest.mark.parametrize("stage", [Stages.train, Stages.test])
def test_train_datamodule(stage):
    # create a qsar datamodule
    fw = Frameworks.torch
    task = "one-shot-learning"
    dm: SiameseDataModule = get_datamodule(
        faces_siamese_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup(stage)

    # get dataloader by stage
    dl = dm.get_stage_dataloader(stage)
    assert dl is not None
