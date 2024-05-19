# local modules
from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules import PandasDataModule
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import house_prices_datamodule_cfg_w_target, \
regrhouse_prices_datamodule_cfg_w_target,\
clusthouse_prices_datamodule_cfg_w_target


def test_classsave_preds(tmp_path):
    # create a house price dm
    fw = Frameworks.sklearn
    task = "table-regression"
    dm: PandasDataModule = get_datamodule(
        house_prices_datamodule_cfg_w_target, fw, task=task
    )
    # for stage train
    stage = Stages.train
    # get target col values
    df = dm.get_stage_dataloader(stage)
    df = dm.get_stage_dataloader(Stages.test)
    y = df["y"]
    # pass "preds", stage, path to the function
    dm.save_preds(y, stage, tmp_path)
    # check if a file has been created
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name in [
        "prediction.csv",
        "regression.csv",
        "clustering.csv",
    ]

def test_regrsave_preds(tmp_path):
    # create a house price dm
    fw = Frameworks.sklearn
    task = "table-regression"
    dm: PandasDataModule = get_datamodule(
        regrhouse_prices_datamodule_cfg_w_target, fw, task=task
    )
    # for stage train
    stage = Stages.train
    # get target col values
    df = dm.get_stage_dataloader(stage)
    df = dm.get_stage_dataloader(Stages.test)
    y = df["y"]
    # pass "preds", stage, path to the function
    dm.save_preds(y, stage, tmp_path)
    # check if a file has been created
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name in [
        "prediction.csv",
        "regression.csv",
        "clustering.csv",
    ]


def test_clustsave_preds(tmp_path):
    # create a house price dm
    fw = Frameworks.sklearn
    task = "table-clustering"
    dm: PandasDataModule = get_datamodule(
        clusthouse_prices_datamodule_cfg_w_target, fw, task=task
    )
    # for stage train
    stage = Stages.train
    # get target col values
    df = dm.get_stage_dataloader(stage)
    df = dm.get_stage_dataloader(Stages.test)
    y = df["y"]
    # pass "preds", stage, path to the function
    dm.save_preds(y, stage, tmp_path)
    # check if a file has been created
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name in [
        "prediction.csv",
        "regression.csv",
        "clustering.csv",
    ]