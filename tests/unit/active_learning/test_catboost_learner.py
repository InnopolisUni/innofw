import os
import shutil

from innofw.constants import Frameworks
from innofw.core import InnoModel
from innofw.core.models.catboost_adapter import CatBoostAdapter
from innofw.core.active_learning.learners import CatBoostActiveLearner
from innofw.utils.framework import get_datamodule
from innofw.utils.framework import get_model
from innofw.utils.getters import get_trainer_cfg, get_log_dir, get_a_learner
from tests.fixtures.config.datasets import house_prices_datamodule_cfg_w_target
from tests.fixtures.config.models import catboost_with_uncertainty_cfg_w_target, catboost_cfg_w_target
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
import hydra


def test_catboost_active_learner_creation():
    model = get_model(
        catboost_with_uncertainty_cfg_w_target, base_trainer_on_cpu_cfg
    )
    task = "table-regression"
    datamodule = get_datamodule(
        house_prices_datamodule_cfg_w_target, Frameworks.catboost, task=task
    )

    sut = CatBoostActiveLearner(
        model=model,
        datamodule=datamodule,
    )

    assert sut is not None


def test_adapter():
    os.makedirs('./tmp', exist_ok=True)
    model = get_model(
        catboost_cfg_w_target, base_trainer_on_cpu_cfg
    )
    adapter = CatBoostAdapter(model, './tmp')
    assert adapter.is_suitable_model(adapter.model)

    adapter.log_results({'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0})
    adapter.prepare_metrics(metrics=[{'_target_': 'sklearn.metrics.mean_squared_error'}])

    datamodule = get_datamodule(
        house_prices_datamodule_cfg_w_target, Frameworks.catboost, task="table-regression"
    )

    adapter._train(datamodule)
    adapter._test(datamodule)

    try:
        shutil.rmtree('./tmp')
    except Exception as e:
        print(e)