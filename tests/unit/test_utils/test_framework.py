import omegaconf
import pytest
from omegaconf import DictConfig
from torchmetrics import F1Score

from innofw.constants import Frameworks
from innofw.utils.framework import get_obj, get_callbacks, get_datamodule_concat


def test_empty_obj_creation():
    assert get_obj(DictConfig({"sample": {"k": "v"}}), name="sample", task=None,
                   framework=None) is None


@pytest.mark.parametrize(["cfg", "task", "framework", "target"], [
    [DictConfig({"callbacks": {}}), "none", Frameworks.none, None],
    [DictConfig({
        "callbacks": {"task": ["table-clustering"],
                      "implementations": {"sklearn":
                                              {"silhouette_score":
                                                   {"_target_": "sklearn.metrics.silhouette_score"}
                                               }
                                          }}}), "table-clustering", Frameworks.sklearn,
        [{"_target_": "sklearn.metrics.silhouette_score"}]],
    [DictConfig({
        "callbacks": {"task": ["multiclass-image-segmentation"],
                      "implementations": {"torch":
                                              {"f1":
                                                   {"_target_": "torchmetrics.F1Score",
                                                    "task": "multiclass", "num_classes": 10}
                                               }
                                          }}}), "multiclass-image-segmentation", Frameworks.torch,
        [F1Score(task="multiclass", num_classes=10)]]
])
def test_get_callbacks(cfg, task, framework, target):
    assert get_callbacks(cfg, task, framework) == target


@pytest.mark.skip(reason="issue with downloading")
def test_get_datamodule_concat():
    cfg = dict()
    for key, path in zip(
            ["first", "second"],
            ["config/datasets/detection_lep.yaml", "config/datasets/detection_lep_insplad.yaml"]
    ):
        conf = omegaconf.OmegaConf.load(path)
        dict_conf = omegaconf.OmegaConf.to_container(conf, resolve=True)
        cfg[key] = dict_conf

    framework = Frameworks.torch
    task = ["image-detection"]
    assert 0, get_datamodule_concat(cfg, framework, task)

# def test_get_datamodule_concat_wrong_task():
# def test_get_datamodule_concat_wrong_frameworks():
