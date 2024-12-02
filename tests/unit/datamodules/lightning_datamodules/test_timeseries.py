import os
import shutil
import pytest

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.lightning_datamodules.anomaly_detection_timeseries_dm import (
    TimeSeriesLightningDataModule,
)
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import anomaly_detection_timeseries_datamodule_cfg_w_target

# local modules


def test_smoke():
    # create a datamodule
    os.makedirs('./tmp', exist_ok=True)

    fw = Frameworks.torch
    task = "anomaly-detection-timeseries"
    dm: TimeSeriesLightningDataModule = get_datamodule(
        anomaly_detection_timeseries_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup(Stages.train)

    assert dm.train is not None
    assert dm.test is not None
    assert dm.train_dataloader() is not None
    assert dm.test_dataloader() is not None
    assert dm.val_dataloader is not None

    dm.save_preds(preds=[0 for _ in range(10)], stage=Stages.train, dst_path='./tmp')
    try:
        shutil.rmtree('./tmp')
    except Exception as e:
        print(e)


@pytest.mark.parametrize("stage", [Stages.train, Stages.test])
def test_train_datamodule(stage):
    os.makedirs('./tmp', exist_ok=True)
    # create  datamodule
    fw = Frameworks.torch
    task = "anomaly-detection-timeseries"
    dm: TimeSeriesLightningDataModule = get_datamodule(
        anomaly_detection_timeseries_datamodule_cfg_w_target, fw, task=task
    )
    assert dm is not None

    # initialize train and test datasets
    dm.setup(stage)
    # get dataloader by stage
    dl = dm.get_stage_dataloader(stage)
    assert dl is not None
    try:
        shutil.rmtree('./tmp')
    except Exception as e:
        print(e)