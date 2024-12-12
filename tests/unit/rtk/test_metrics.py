import os

import numpy as np
import pandas as pd
import pytest

from innofw.utils.data_utils.rtk.CT_hemorrhage_metrics import compute_metrics
from innofw.utils.data_utils.rtk.lungs_description_metrics import (
    calculate_lungs_metrics,
)


@pytest.fixture
def generate_test_data(tmpdir):
    # data generation for y_pred и y_gt
    pred_data = pd.DataFrame(
        {"y": ["Патология", "Норма", "Норма", "Патология", "необходимо дообследование"]}
    )

    gt_data = pd.DataFrame(
        {
            "decision": [
                "Патология ",
                "Норма",
                "необходимо дообследование",
                "Патология",
                "Норма",
            ]
        }
    )

    pred_path = os.path.join(tmpdir, "predictions.csv")
    pred_data.to_csv(pred_path, index=False)

    gt_path = os.path.join(tmpdir, "ground_truth.csv")
    gt_data.to_csv(gt_path, index=False)

    return pred_path, gt_path


def test_lungs_metrics(generate_test_data, capsys):
    pred_path, gt_path = generate_test_data

    calculate_lungs_metrics(pred_path, gt_path)

    captured = capsys.readouterr()

    assert "Отчет по метрикам" in captured.out
    assert "Матрица ошибок" in captured.out
    assert "Патология" in captured.out
    assert "Норма" in captured.out
    assert "необходимо дообследование" in captured.out


def test_compute_metrics():
    gt_boxes = [[0, 0, 2, 2], [2, 2, 4, 4]]
    # pr_boxes = [[1, 1, 3, 3], [3, 3, 5, 5]]

    pr_boxes_same = [[0, 0, 2, 2], [2, 2, 4, 4]]
    metrics = compute_metrics(gt_boxes, pr_boxes_same, iou_threshold=0.5)

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["mean_iou"] == pytest.approx(1.0)

    pr_boxes_empty = []
    metrics = compute_metrics(gt_boxes, pr_boxes_empty, iou_threshold=0.5)

    assert metrics["precision"] == 0
    assert metrics["recall"] == 0
    assert metrics["mean_iou"] == 0
