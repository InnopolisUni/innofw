#
from typing import Callable

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import rand_score
from sklearn.metrics import v_measure_score

from innofw.core.metrics import Metric

#
#


@pytest.mark.parametrize(
    ["metric_func"],
    [
        [rand_score],
        [adjusted_rand_score],
        [adjusted_mutual_info_score],
        [homogeneity_score],
        [completeness_score],
        [v_measure_score],
        # [silhouette_score],
    ],
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
