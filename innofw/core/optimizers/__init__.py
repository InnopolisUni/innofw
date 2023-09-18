"""
This package is part of our framework's CORE, which is meant to give flexible support for optimizers from different
libraries and frameworks via common abstract wrapper, currently it has support for:
- pytorch

Optimizers are algorithms or methods used to change the attributes of your neural network such as weights and learning
rate in order to reduce the losses.
"""
__all__ = ["get_optim_adapter", "Optimizer"]

#
import os
import importlib

#
import torch.nn as nn
from torch.optim import Optimizer as TorchOptim

#
from innofw.core.optimizers.base import BaseOptimizerAdapter


def factory_method(name):
    return __OPTIM_ADAP_DICT__[name]


__OPTIM_ADAP_DICT__ = dict()


def get_optim_adapter(optimizer):
    suitable_optimizers = [
        optim_adapter
        for optim_adapter in __OPTIM_ADAP_DICT__.values()
        if optim_adapter.is_suitable_input(optimizer)
    ]
    if len(suitable_optimizers) == 0:
        raise NotImplementedError()
    elif len(suitable_optimizers):
        return suitable_optimizers[0](optimizer)


def register_optimizers_adapter(name):
    def register_function_fn(cls):
        if name in __OPTIM_ADAP_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, BaseOptimizerAdapter):
            raise ValueError(
                "Class %s is not a subclass of %s" % (cls, BaseOptimizerAdapter)
            )
        __OPTIM_ADAP_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[: file.find(".py")]
        module = importlib.import_module("innofw.core.optimizers." + module_name)


class Optimizer(TorchOptim):
    """
    Class provides same interface for different optimizers by utilizing adapters

    Methods
    -------
    step(x)
        updates a model parameters
    param_groups()
        returns model's parameters
    """

    def __init__(self, optimizer=None):
        if optimizer is not None:
            self.optim = get_optim_adapter(optimizer)

    def step(self, *args, **kwargs):
        return self.optim.step(*args, **kwargs)

    @property
    def param_groups(self):
        return self.optim.param_groups

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def state(self):
        return self.optim.state

    @property
    def params(self):  # iterable
        return self.optim.params

    @property
    def defaults(self) -> dict:
        return self.optim.defaults
