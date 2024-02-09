from innofw.constants import Frameworks
from innofw.core.active_learning.datamodule import ActiveDataModule, DataModuleI, get_active_datamodule
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import qm9_datamodule_cfg_w_target
from hydra.core.global_hydra import GlobalHydra


def test_active_datamodule_creation():
    GlobalHydra.instance().clear()
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.catboost, task=task
    )
    sut = ActiveDataModule(datamodule=datamodule)
    sut.setup()
    assert sut is not None


def test_update_indices():
    import numpy as np
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.catboost, task=task
    )
    sut = ActiveDataModule(datamodule=datamodule)

    indices_update = np.array([2,3])
    sut.train_idxs = np.array([0,1,2])
    sut.pool_idxs = np.array([10,20,30,40,50])
    sut.update_indices(indices_update)
    assert np.array_equal(sut.train_idxs, np.array([0,1,2,30,40]))
    assert np.array_equal(sut.pool_idxs, np.array([10,20,50]))

def test_train_dataloader():
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.catboost, task=task
    )
    sut = ActiveDataModule(datamodule=datamodule)
    train_dataloader = sut.train_dataloader()
    assert 'x' in train_dataloader
    assert 'y' in train_dataloader
    assert train_dataloader['x'] is not None
    assert train_dataloader['y'] is not None

def test_test_dataloader():
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.catboost, task=task
    )
    sut = ActiveDataModule(datamodule=datamodule)
    assert sut.test_dataloader() is not None

def test_pool_dataloader():
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.catboost, task=task
    )
    sut = ActiveDataModule(datamodule=datamodule)
    pool_dataloader = sut.pool_dataloader()
    assert 'x' in pool_dataloader
    assert pool_dataloader['x'] is not None

def test_get_active_datamodule():
    task = "qsar-regression"
    datamodule = get_datamodule(
        qm9_datamodule_cfg_w_target, Frameworks.catboost, task=task
    )
    dm = get_active_datamodule(datamodule)

    assert dm is not None
    assert dm.__class__ is ActiveDataModule
