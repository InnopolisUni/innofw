from time import sleep

from omegaconf import DictConfig
import pytest

from innofw import InnoModel
from innofw.pipeline import run_pipeline
from innofw.utils.framework import get_model, get_datamodule, map_model_to_framework

from tests.utils import get_test_folder_path
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg
from tests.fixtures.config.models import linear_regression_cfg_w_target
from tests.fixtures.config.datasets import house_prices_datamodule_cfg_w_target


@pytest.mark.parametrize(
    ["model_cfg", "dm_cfg", "trainer_cfg", "task", "ckpt_path"],
    [
        [
            linear_regression_cfg_w_target,
            house_prices_datamodule_cfg_w_target,
            base_trainer_on_cpu_cfg,
            "table-regression",
            None,
        ],
        [
            linear_regression_cfg_w_target,
            house_prices_datamodule_cfg_w_target,
            base_trainer_on_cpu_cfg,
            "table-regression",
            str(
                (
                    get_test_folder_path()
                    / "weights/regression_house_prices/lin_reg.pickle"
                ).resolve()
            ),
        ],
    ],
)
def test_linear_regression_training(
    model_cfg, dm_cfg, trainer_cfg, task, ckpt_path, tmp_path
):
    cfg = DictConfig(
        {
            "models": model_cfg,
            "datasets": dm_cfg,
            "trainer": trainer_cfg,
            "task": task,
            "batch_size": 3,
            "epochs": 1,
            "accelerator": "cpu",
            "project": "lin_reg_training",
            "experiment": "something",
            "ckpt_path": ckpt_path,
            "weights_path": ckpt_path,
            "experiment_name": "something"
        }
    )
    assert len(list(tmp_path.iterdir())) == 0, "log root should be empty"
    run_pipeline(cfg, train=True, predict=False, test=False, log_root=tmp_path)

    files = list(tmp_path.iterdir())
    assert (
        len(files) == 1 and files[0].is_dir()
    ), "log root should contain only one folder"

    checkpoint_files = list(tmp_path.rglob("model.pickle"))
    assert (
        len(checkpoint_files) == 1 and checkpoint_files[0].parent.name == "checkpoints"
    ), "there should a checkpoint file and it should be in a folder named `checkpoint`"


@pytest.mark.parametrize(
    ["model_cfg", "dm_cfg", "trainer_cfg", "task", "ckpt_path"],
    [
        [
            linear_regression_cfg_w_target,
            house_prices_datamodule_cfg_w_target,
            base_trainer_on_cpu_cfg,
            "table-regression",
            None,
        ],
        [
            linear_regression_cfg_w_target,
            house_prices_datamodule_cfg_w_target,
            base_trainer_on_cpu_cfg,
            "table-regression",
            (
                get_test_folder_path()
                / "weights/regression_house_prices/lin_reg.pickle"
            ).resolve(),
        ],
    ],
)
def test_linear_regression_multiple_training(
    model_cfg, dm_cfg, trainer_cfg, task, ckpt_path, tmp_path
):
    model = get_model(model_cfg, trainer_cfg)
    datamodule = get_datamodule(dm_cfg, framework=map_model_to_framework(model), task=task)
    log_dir = tmp_path / "logs"
    InnoModel(model=model, log_dir=log_dir).train(datamodule, ckpt_path)
