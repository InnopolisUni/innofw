from typing import Callable

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from innofw.core.metrics import Metric
from innofw.core.metrics.custom_metrics.metrics import Accuracy
from innofw.core.metrics.custom_metrics.metrics import F1Score
from innofw.core.metrics.custom_metrics.metrics import Precision
from innofw.core.metrics.custom_metrics.metrics import Recall

#
#


@pytest.mark.parametrize(
    ["metric_func", "sklearn_implementation"],
    [
        [F1Score, f1_score],
        [Accuracy, accuracy_score],
        [Recall, recall_score],
        [Precision, precision_score],
    ],
)
def test_sklearn_metrics_adapter_on_custom(
    metric_func: Callable, sklearn_implementation: Callable
):
    pred = np.array([0, 1, 0])
    label = np.array([1, 0, 1])
    # compute score
    metric = Metric(metric_func)
    score = metric(pred, label)
    # check result
    assert score is not None
    assert isinstance(score, np.float)
    assert score == f1_score(pred, label)
