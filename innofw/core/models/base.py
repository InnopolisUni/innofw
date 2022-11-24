from abc import ABC, abstractmethod
from pathlib import Path

from innofw.constants import Stages


class BaseModelAdapter(ABC):
    """
    Base Adapter for working with models
    ...

    Attributes
    ----------
    model : nn.Module
        model to train
    log_dir : Path
        a path to logging directory
    ckpt_handler :
        loads model from checkpoints

    Methods
    -------
    predict(x):
        returns result of prediction and saves them
    train(self, datamodule, ckpt_path=None):
        trains model and saves checkpoints

    """
    @staticmethod
    @abstractmethod
    def is_suitable_model(model) -> bool:
        pass

    def __init__(self, model, log_dir, ckpt_handler=None):
        self.model = model
        self.log_dir = Path(log_dir)
        self.ckpt_handler = ckpt_handler

    def predict(self, datamodule, ckpt_path=None):
        if ckpt_path is not None:
            self.model = self.ckpt_handler.load_model(self.model, ckpt_path)
        result = self._predict(datamodule)
        return datamodule.save_preds(
            result, stage=Stages.predict, dst_path=self.log_dir
        )

    @abstractmethod
    def _predict(self, data):
        pass

    @abstractmethod
    def _test(self, data):
        pass

    def train(self, datamodule, ckpt_path=None):
        if ckpt_path is not None:
            self.model = self.ckpt_handler.load_model(self.model, ckpt_path)
        result = self._train(datamodule)
        self.save_ckpt(self.model)
        return result

    def test(self, datamodule, ckpt_path=None):
        if ckpt_path is not None:
            self.model = self.ckpt_handler.load_model(self.model, ckpt_path)
        result = self._test(datamodule)
        self.save_ckpt(self.model)
        return result

    @abstractmethod
    def _train(self, data):
        pass

    def set_stop_params(self, stop_param):
        pass

    def set_checkpoint_save(self, weights_path, weights_freq, project, experiment):
        pass

    def save_ckpt(self, model):
        self.ckpt_handler.save_ckpt(model, self.log_dir, create_default_folder=True)

    def load_ckpt(self, path):
        return self.ckpt_handler.load_ckpt(path)
