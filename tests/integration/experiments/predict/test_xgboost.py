from omegaconf import DictConfig
import pytest

from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg
from tests.fixtures.config.models import xgbregressor_cfg_w_target
from tests.fixtures.config.datasets import house_prices_datamodule_cfg_w_target
from innofw.pipeline import run_pipeline
from tests.utils import get_test_data_folder_path, get_test_folder_path

house_prices_predict_datamodule_cfg_w_target = (
    house_prices_datamodule_cfg_w_target.copy()
)

house_prices_predict_datamodule_cfg_w_target["infer"] = {
    "source": str(
        get_test_data_folder_path() / "tabular/regression/house_prices/test/test.csv"
    )
}


@pytest.mark.parametrize(
    ["model_cfg", "dm_cfg", "trainer_cfg", "task", "ckpt_path"],
    [
        [
            xgbregressor_cfg_w_target,
            house_prices_predict_datamodule_cfg_w_target,
            base_trainer_on_cpu_cfg,
            "table-regression",
            str(
                (
                    get_test_folder_path()
                    / "weights/regression_house_prices/xgboost_regression_best.pkl"
                ).resolve()
            ),
        ],
    ],
)
def test_xgboost_prediction(model_cfg, dm_cfg, trainer_cfg, task, ckpt_path):
    cfg = DictConfig(
        {
            "models": model_cfg,
            "datasets": dm_cfg,
            "trainer": trainer_cfg,
            "task": task,
            "batch_size": 3,
            "epochs": 1,
            "accelerator": "cpu",
            "project": "xgboost_prediction",
            "experiment": "something",
            "ckpt_path": ckpt_path,
            "weights_path": ckpt_path,
        }
    )
    run_pipeline(cfg, train=False, predict=True)
