#
from typing import Callable

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from innofw.core.metrics import Metric

#
#


@pytest.mark.parametrize(
    ["metric_func"],
    [[accuracy_score], [f1_score], [precision_score], [recall_score]],
)
def test_sklearn_metrics_adapter(metric_func: Callable):
    pred = np.array([0, 1, 0])
    label = np.array([1, 0, 1])
    # compute score
    metric = Metric(metric_func)
    score = metric(pred, label)
    # check result
    assert score is not None
    assert isinstance(score, np.float)
