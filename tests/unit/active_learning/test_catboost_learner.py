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

    for i in range(3):
        try:
            shutil.rmtree('./tmp')
            break
        except:
            pass

# def test_base_learner_run():
#     model = get_model(
#         catboost_with_uncertainty_cfg_w_target, base_trainer_on_cpu_cfg
#     )
#     #GR_230822_ASdw31ga_catboost_industry_data
#     # /workspace/innofw/config/experiments/regression/GR_230822_ASdw31ga_catboost_industry_data.yaml
#     # /workspace/innofw/config/train.yaml

#     model = InnoModel(model, log_dir="./logs/test/test1/")
#     task = "table-regression"
#     datamodule = get_datamodule(
#         house_prices_datamodule_cfg_w_target, Frameworks.catboost, task=task
#     )

#     sut = CatBoostActiveLearner(
#         model=model,
#         datamodule=datamodule,
#     )

#     sut.run(ckpt_path='/workspace/innofw/tests/weights/catboost_industry_data/model.pickle')

#     assert sut is not None

# from innofw.utils.loggers import setup_clear_ml, setup_wandb
# def test_base_learner_run_empty_pool():
#     # if not config.get("experiment_name"):
#     #     hydra_cfg = HydraConfig.get()
#     #     experiment_name = OmegaConf.to_container(hydra_cfg.runtime.choices)[
#     #         "experiments"
#     #     ]
#     #     config.experiment_name = experiment_name

#     config = OmegaConf.load('/workspace/innofw/config/experiments/regression/GR_230822_ASdw31ga_catboost_industry_data.yaml')
#     print(config)
#     setup_clear_ml(config)
#     setup_wandb(config)
#     model = get_model(
#         catboost_with_uncertainty_cfg_w_target, base_trainer_on_cpu_cfg
#     )   


#      # wrap the model
#     model_params = {
#         "model": model,
#         "log_dir": "./logs/test/test1/",
#     }
#     model = InnoModel(**model_params)


#     task = "table-regression"
#     datamodule = get_datamodule(
#         house_prices_datamodule_cfg_w_target, Frameworks.catboost, task=task
#     )

#     datamodule.pool_idxs = []
#     a_learner = get_a_learner(config, model, datamodule)

#     # sut = CatBoostActiveLearner(
#     #     model=model,
#     #     datamodule=datamodule,
#     # )

#     ret = a_learner.run(datamodule, ckpt_path='/workspace/innofw/tests/weights/catboost_industry_data/model.pickle')

#     assert ret is None