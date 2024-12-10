import os

# import numpy as np
import pandas as pd
import pytest

from innofw.utils.data_utils.rtk.CT_hemorrhage_metrics import (
    compute_iou,
    calculate_iou_bbox,
    compute_metrics,
    np,
)
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


def test_compute_iou():
    mask1 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]])
    mask2 = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])

    expected_iou = 3 / 5  # intersection=3, union=5
    assert compute_iou(mask1, mask2) == pytest.approx(expected_iou)

    mask_same = np.array([[1, 1], [1, 1]])
    assert compute_iou(mask_same, mask_same) == 1.0

    mask_no_intersection = np.array([[0, 0], [0, 0]])
    assert compute_iou(mask_no_intersection, mask_same) == 0


def test_calculate_iou_bbox():
    box1 = [0, 0, 2, 2]
    box2 = [1, 1, 3, 3]

    expected_iou = 1 / 7  # intersection=1, union=7
    assert calculate_iou_bbox(box1, box2) == pytest.approx(expected_iou)

    # Тест, когда bounding boxes полностью совпадают
    assert calculate_iou_bbox(box1, box1) == 1.0

    # Тест, когда нет пересечения
    box_no_intersection = [3, 3, 5, 5]
    assert calculate_iou_bbox(box1, box_no_intersection) == 0


def test_compute_metrics():
    gt_boxes = [[0, 0, 2, 2], [2, 2, 4, 4]]
    # pr_boxes = [[1, 1, 3, 3], [3, 3, 5, 5]]

    pr_boxes_same = [[0, 0, 2, 2], [2, 2, 4, 4]]
    metrics = compute_metrics(gt_boxes, pr_boxes_same, iou_threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["mean_iou"] == pytest.approx(1.0)

    pr_boxes_empty = []
    metrics = compute_metrics(gt_boxes, pr_boxes_empty, iou_threshold=0.5)

    assert metrics["precision"] == 0
    assert metrics["recall"] == 0
    assert metrics["mean_iou"] == 0
