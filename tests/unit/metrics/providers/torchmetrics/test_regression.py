#
from typing import Callable

#
import pytest
import torch
import numpy as np
from torchmetrics.functional import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

#
from innofw.core.metrics import Metric


@pytest.mark.parametrize(
    ["metric_func"],
    [
        [r2_score],
        [mean_absolute_error],
        [mean_squared_error],
    ],
)
def test_torch_metrics_adapter(metric_func: Callable):
    # working with tensors
    pred = torch.tensor([0.0, 1.0, 0.0])
    label = torch.tensor([1, 0, 1])
    # compute score
    metric = Metric(metric_func)
    score = metric(pred, label)
    # check result
    assert score is not None
    assert isinstance(score, torch.FloatTensor)
