"""
This package is part of our framework's CORE, which is meant to give flexible support for models from different
libraries and frameworks via common abstract wrapper, currently it has support for:
- pytorch
- segmentation_models_pytorch
- sklearn

Also it has a pseudocode folder.

Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions
or decisions without being explicitly programmed to do so on unseen but similar data sets.
"""

__all__ = ["get_model_adapter", "register_models_adapter", "InnoModel"]
#
import os
import importlib

#
import torch.nn as nn

#
from innofw.core.models.base import BaseModelAdapter


def factory_method(name):
    return __MODEL_ADAP_DICT__[name]


__MODEL_ADAP_DICT__ = dict()


def get_model_adapter(model, *args, **kwargs):
    suitable_adapters = [
        aug_adapter
        for aug_adapter in __MODEL_ADAP_DICT__.values()
        if aug_adapter.is_suitable_model(model)
    ]
    if len(suitable_adapters) == 0:
        raise NotImplementedError()
    elif len(suitable_adapters):
        return suitable_adapters[0](model, *args, **kwargs)


def register_models_adapter(name):
    def register_function_fn(cls):
        if name in __MODEL_ADAP_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, BaseModelAdapter):
            raise ValueError(
                "Class %s is not a subclass of %s" % (cls, BaseModelAdapter)
            )
        __MODEL_ADAP_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith("_adapter.py") and not file.startswith("_"):
        module_name = file[: file.find(".py")]
        module = importlib.import_module("innofw.core.models." + module_name)


class InnoModel(nn.Module):  # todo: refactor
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = get_model_adapter(model, *args, **kwargs)

    def forward(self, x):
        return self.model(x)  # todo: check if it works. seemingly not

    def predict(self, datamodule, ckpt_path=None):
        return self.model.predict(datamodule, ckpt_path)

    def train(self, datamodule, ckpt_path=None):
        return self.model.train(datamodule, ckpt_path)

    def test(self, datamodule, ckpt_path=None):
        return self.model.test(datamodule, ckpt_path)

    def set_stop_params(self, stop_param):
        return self.model.set_stop_params(stop_param)

    def set_checkpoint_save(self, weights_path, weights_freq, project, experiment):
        return self.model.set_checkpoint_save(
            weights_path, weights_freq, project, experiment
        )

    def save_ckpt(self, model):  # todo: refactor, should not receive an argument
        return self.model.save_ckpt(model)

    def load_ckpt(self, path):
        return self.model.load_ckpt(path)

    def __repr__(self):
        return str(self.model)
