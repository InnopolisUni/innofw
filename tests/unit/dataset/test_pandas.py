from omegaconf import DictConfig

from innofw.constants import Frameworks
from innofw.utils.framework import get_datamodule, get_obj


def test_classification_pandas():
    cfg = DictConfig(
        {
            "datasets": {
                "name": "something",
                "description": "something",
                "markup_info": "something",
                "date_time": "something",
                "task": ["table-classification"],
                "train": {"source": "tests/data/tabular/bmi/bmi_prep.csv"},
                "test": {"source": "tests/data/tabular/bmi/bmi_prep.csv"},
                "target_col": "Index",
            },
        }
    )
    task = "table-classification"
    framework = Frameworks.xgboost
    dm = get_datamodule(cfg.datasets, framework)
    assert dm

    train_data = dm.train_dataloader()
    assert train_data.get("x") is not None and train_data.get("y") is not None

    test_data = dm.test_dataloader()
    assert test_data.get("x") is not None and test_data.get("y") is not None
