"""
This package is part of our framework's CORE, which is meant to give flexible support for losses from different
libraries and frameworks via common abstract wrapper, currently it has support for:
- pytorch


Loss is the penalty for a bad prediction. That is, loss is a number indicating how bad the model's prediction was
on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater.
"""

__all__ = ["get_loss_adapter", "Loss"]
#
import os
import importlib

#
import torch.nn as nn

#
from innofw.core.losses.base import BaseLossAdapter


def factory_method(name):
    return __LOSS_ADAP_DICT__[name]


__LOSS_ADAP_DICT__ = dict()


def get_loss_adapter(loss):
    suitable_losses = [
        loss_adapter
        for loss_adapter in __LOSS_ADAP_DICT__.values()
        if loss_adapter.is_suitable_input(loss)
    ]
    if len(suitable_losses) == 0:
        raise NotImplementedError()
    elif len(suitable_losses):
        return suitable_losses[0](loss)


def register_losses_adapter(name):
    def register_function_fn(cls):
        if name in __LOSS_ADAP_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, BaseLossAdapter):
            raise ValueError(
                "Class %s is not a subclass of %s" % (cls, BaseLossAdapter)
            )
        __LOSS_ADAP_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[: file.find(".py")]
        module = importlib.import_module("innofw.core.losses." + module_name)


class Loss(nn.Module):
    """
    Inner loss adapter
    """

    def __init__(self, losses):
        super().__init__()
        self.loss = get_loss_adapter(losses)

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    def __repr__(self):
        return str(self.loss)
