#
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from innofw.utils.checkpoint_utils import add_metadata2model
from innofw.utils.checkpoint_utils import CheckpointHandler
from innofw.utils.checkpoint_utils import load_metadata
from innofw.utils.checkpoint_utils import PickleCheckpointHandler
from innofw.utils.checkpoint_utils.torch_checkpoint_handler import (
    TorchCheckpointHandler,
)
from tests.fixtures.models import catboot_cls_model
from tests.fixtures.models import lightning_model
from tests.fixtures.models import sklearn_reg_model
from tests.fixtures.models import torch_model
from tests.fixtures.models import xgb_reg_model

#
#


def run_checkpoint_handler_tests(file_path):
    assert file_path.exists(), "file should be existing"

    # 1. adding the metadata
    metadata = {"score": 0.7, "name": "linear regression"}
    add_metadata2model(file_path, metadata, check_schema=False)
    file_metadata = load_metadata(file_path)

    assert (
        metadata.items() <= file_metadata.items()
    ), "new keys should be appended to the model metadata"

    # 2. appending new metadata
    additional_metadata = {
        "_target_": "innoframework.core.models.sklearn.linear_regression.LinearRegression"
    }
    add_metadata2model(file_path, additional_metadata, check_schema=False)
    file_metadata = load_metadata(file_path)

    assert (
        additional_metadata.items() <= file_metadata.items()
    ), "new keys should be appended to the model metadata"

    # 3. overwrite the metadata
    new_metadata = {"score": 0.76}
    add_metadata2model(file_path, new_metadata, check_schema=False)
    file_metadata = load_metadata(file_path)

    assert (
        new_metadata.items() <= file_metadata.items()
    ), "field value should be present and overwritten"
    assert (
        file_metadata["name"] == metadata["name"]
    ), "old fields should not be removed"


@pytest.mark.parametrize(
    ["model", "checkpoint_handler"],
    [
        [
            sklearn_reg_model,
            PickleCheckpointHandler(),
        ],
        [
            xgb_reg_model,
            PickleCheckpointHandler(),
        ],
        [
            torch_model,
            TorchCheckpointHandler(),
        ],
        [
            lightning_model,
            TorchCheckpointHandler(),
        ],
        [
            catboot_cls_model,
            PickleCheckpointHandler(),
        ],
    ],
)
def test_checkpoint_metadata(
    model: Any,
    checkpoint_handler: CheckpointHandler,
    tmp_path: Path,
):
    # 1. add some metadata
    # assert is_path_empty(tmp_path)
    # save_path = checkpoint_handler.save_ckpt(model, tmp_path)
    # run_checkpoint_handler_tests(save_path)

    # 2. add metadata with schema validation
    save_path = checkpoint_handler.save_ckpt(model, tmp_path)
    run_checkpoint_handler_tests_w_schema_validation(save_path)


def run_checkpoint_handler_tests_w_schema_validation(file_path):
    assert file_path.exists(), "file should be existing"

    # 1. adding the metadata
    metadata = {
        "name": "some model",
        "_target_": "some.path.to.model",
        "data": "some/path/to/data",
        "metrics": {"recall": 0.0, "precision": 1.0},
        "description": "some description",
        "weights": "s3://some.com/path/to/the/model",
    }
    add_metadata2model(file_path, metadata, check_schema=True)
    file_metadata = load_metadata(file_path)

    assert isinstance(file_metadata, dict)
    assert (
        metadata.items() <= file_metadata.items()
    ), "new keys should be appended to the model metadata"

    wrong_metadata = {"name": "some name"}

    with pytest.raises(ValidationError):
        add_metadata2model(file_path, wrong_metadata, check_schema=True)


def test_wrong_model_saving():
    handler = TorchCheckpointHandler()
    with pytest.raises(Exception):
        handler.save_ckpt(None, "./tmp")


def test_ckpt_conversion(tmp_path: Path):
    checkpoint_handler = PickleCheckpointHandler()
    save_path = checkpoint_handler.save_ckpt(sklearn_reg_model, tmp_path)
    run_checkpoint_handler_tests_w_schema_validation(save_path)
    checkpoint_handler.convert_to_regular_ckpt(save_path, save_path, inplace=False)