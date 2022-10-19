from typing import Any
from pathlib import Path
from typing import Callable

#
import torch
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#
from innofw.utils import is_path_empty
from tests.fixtures.models import (
    sklearn_reg_model,
    xgb_reg_model,
    torch_model,
    lightning_model,
    catboot_cls_model,
)
from innofw.utils.checkpoint_utils import (
    CheckpointHandler,
    PickleCheckpointHandler,
    TorchCheckpointHandler,
)


def sklearn_regression_model_functionality_check(model):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred is not None
    assert len(y_pred) == len(X_test)


def torch_classification_model_functionality_check(model):
    img = torch.rand(1, 28 * 28)
    pred = model(img)
    assert pred is not None


@pytest.mark.parametrize(
    ["model", "checkpoint_handler", "model_functionality_tester"],
    [
        [
            sklearn_reg_model,
            PickleCheckpointHandler(),
            sklearn_regression_model_functionality_check,
        ],
        [
            xgb_reg_model,
            PickleCheckpointHandler(),
            sklearn_regression_model_functionality_check,
        ],
        [
            torch_model,
            TorchCheckpointHandler(),
            torch_classification_model_functionality_check,
        ],
        [
            lightning_model,
            TorchCheckpointHandler(),
            torch_classification_model_functionality_check,
        ],
        [
            catboot_cls_model,
            PickleCheckpointHandler(),
            sklearn_regression_model_functionality_check,
        ],
    ],
)
def test_checkpoint_saving(
    model: Any,
    checkpoint_handler: CheckpointHandler,
    model_functionality_tester: Callable,
    tmp_path: Path,
):
    assert is_path_empty(tmp_path)
    save_path = checkpoint_handler.save_ckpt(model, tmp_path)
    assert not is_path_empty(tmp_path) and save_path.exists()

    model = checkpoint_handler.load_model(model, save_path)
    assert model is not None
    model_functionality_tester(model)
