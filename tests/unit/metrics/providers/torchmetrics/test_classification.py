#
from typing import Callable

import pytest
import torch
from torchmetrics.functional import accuracy
from torchmetrics.functional import f1_score

from innofw.core.metrics import Metric

#
#


@pytest.mark.parametrize(
    ["metric_func"],
    [
        [accuracy],
        [f1_score],
    ],
)
def test_torch_metrics_adapter(metric_func: Callable):
    # working with tensors
    pred = torch.tensor([0.0, 1.0, 0.0])
    label = torch.tensor([1, 0, 1])
    # compute score
    metric = Metric(metric_func)
    score = metric(pred, label, task="binary")
    # check result
    assert score is not None
    assert isinstance(score, torch.FloatTensor)
