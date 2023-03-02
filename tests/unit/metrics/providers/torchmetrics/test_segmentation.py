# other
from omegaconf import DictConfig
import torch
import pytest

# local
from innofw.constants import Frameworks
from innofw.utils.framework import get_obj


def test_metrics_creation():
    cfg = DictConfig(
        {
            "metrics": {
                "task": ["image-segmentation"],
                "implementations": {
                    "torch": {
                        "F1_score": {
                            "function": {
                                "_target_": "torchmetrics.functional.f1_score",
                                "task": "binary",
                                "num_classes": 2,
                            }
                        }
                    },
                    "meta": {
                        "description": "Set of default metrics for semantic segmentation"
                    },
                },
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.torch
    metrics = get_obj(cfg, "metrics", task, framework)

    assert metrics is not None
    assert not isinstance(metrics, list)

    target = torch.tensor([0, 1, 1, 0, 1, 0])
    preds = torch.tensor([0, 0, 1, 0, 0, 1])  # 1 / (1 + 0.5 * (1 + 2))

    score = metrics(preds, target)
    assert score == torch.tensor(0.4)


def test_multiple_metrics_creation():
    cfg = DictConfig(
        {
            "metrics": {
                "task": ["image-segmentation"],
                "implementations": {
                    "torch": {
                        "F1_score": {
                            "function": {
                                "_target_": "torchmetrics.functional.f1_score",
                                "task": "binary",
                                "num_classes": 2,
                            }
                        },
                        "IOU_score": {
                            "object": {
                                "_target_": "torchmetrics.JaccardIndex",
                                "task": "binary",
                                "num_classes": 2,
                            }
                        },
                        "Precision": {
                            "object": {
                                "_target_": "torchmetrics.Precision",
                                "task": "binary",
                                "num_classes": 2,
                            }
                        },
                        "meta": {
                            "description": "Set of default metrics for semantic segmentation"
                        },
                    },
                },
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.torch
    metrics = get_obj(cfg, "metrics", task, framework)

    assert metrics is not None
    assert isinstance(metrics, list)

    target = torch.tensor([0, 1, 1, 0, 1, 0])
    preds = torch.tensor([0, 0, 1, 0, 0, 1])

    for metric in metrics:
        score = metric(preds, target)
        assert score >= 0


def test_metrics_creation_wrong_task():
    cfg = DictConfig(
        {
            "metrics": {
                "task": ["image-segmentation"],
                "implementations": {
                    "torch": {
                        "F1_score": {
                            "function": {
                                "_target_": "torchmetrics.functional.f1_score",
                                "num_classes": 2,
                            }
                        }
                    },
                    "meta": {
                        "description": "Set of default metrics for semantic segmentation"
                    },
                },
            }
        }
    )
    task = "image-classification"
    framework = "torch"

    with pytest.raises(ValueError):
        metrics = get_obj(cfg, "metrics", task, framework)


def test_metrics_creation_wrong_framework():
    cfg = DictConfig(
        {
            "metrics": {
                "task": ["image-segmentation"],
                "implementations": {
                    "torch": {
                        "F1_score": {
                            "function": {
                                "_target_": "torchmetrics.functional.f1_score",
                                "num_classes": 2,
                            }
                        }
                    },
                    "meta": {
                        "description": "Set of default metrics for semantic segmentation"
                    },
                },
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.sklearn

    with pytest.raises(ValueError):
        metrics = get_obj(cfg, "metrics", task, framework)
