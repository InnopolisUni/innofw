import pytest
from omegaconf import OmegaConf

clear_ml = pytest.importorskip("clearml")
from dataclasses import dataclass

from innofw.utils.loggers import setup_clear_ml


@dataclass
class Cfg:
    def __init__(self):
        class ClearML:
            def __init__(self):
                self.project = "test"
                self.enable = True
                self.task = "test"
                self.queue = None

        self.clear_ml = ClearML()


@dataclass
class Hydra:
    @dataclass
    class Choices:
        choices = None

    runtime = Choices()


class Task:
    def __init__(self):
        pass

    def init(self, *args, **kwargs):
        return self

    def execute_remotely(self, *args, **kwargs):
        return True

    def connect(self, *args, **kwargs):
        return True


def test_clear_ml_task_creation(mocker):
    mocker.patch("clearml.Task.init", return_value=Task().init())
    mocker.patch("clearml.Task.connect", return_value=Task().connect())
    cfg = {
        "clear_ml": {"enable": True, "task": "test", "queue": None},
        "project": "test",
        "experiment_name": "test",
    }
    cfg = OmegaConf.create(cfg)
    task = setup_clear_ml(cfg)
    assert task is not None

    cfg["clear_ml"]["enable"] = False
    task = setup_clear_ml(cfg)
    assert task is None


def test_clear_ml_agent_execution(mocker):
    mocker.patch("clearml.Task.init", return_value=Task().init())
    mocker.patch("clearml.Task.connect", return_value=Task().connect())
    cfg = {
        "clear_ml": {"enable": True, "task": "test", "queue": None},
        "project": "test",
        "experiment_name": "test",
    }
    cfg = OmegaConf.create(cfg)
    cfg["clear_ml"]["queue"] = "no_queue"
    task = setup_clear_ml(cfg)
    assert task is not None
