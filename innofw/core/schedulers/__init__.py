"""
This package is part of our framework's CORE, which is meant to give flexible support for schedulers from different
libraries and frameworks via common abstract wrapper, currently it has support for:
- pytorch

Learning rate schedulers are functions which adjust current learning rate based on epoch count.
"""
__all__ = ["get_sheduler_adapter", "Scheduler"]

#
import os
import importlib

#
import torch.nn as nn

#
from innofw.core.schedulers.base import BaseSchedulerAdapter
from innofw.core.optimizers import Optimizer


def factory_method(name):
    return __SCHEDULER_ADAP_DICT__[name]


__SCHEDULER_ADAP_DICT__ = dict()


def get_sheduler_adapter(scheduler, optimizer, *args, **kwargs):
    suitable_schedulers = [
        scheduler_adapter
        for scheduler_adapter in __SCHEDULER_ADAP_DICT__.values()
        if scheduler_adapter.is_suitable_input(scheduler)
    ]
    if len(suitable_schedulers) == 0:
        raise NotImplementedError()
    elif len(suitable_schedulers):
        return suitable_schedulers[0](scheduler, optimizer, *args, **kwargs)


def register_scheduler_adapter(name):
    def register_function_fn(cls):
        if name in __SCHEDULER_ADAP_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, BaseSchedulerAdapter):
            raise ValueError(
                "Class %s is not a subclass of %s" % (cls, BaseSchedulerAdapter)
            )
        __SCHEDULER_ADAP_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[: file.find(".py")]
        module = importlib.import_module("innofw.core.schedulers." + module_name)


class Scheduler(nn.Module):
    def __init__(self, scheduler, optimizer: Optimizer, *args, **kwargs):
        super().__init__()
        self.scheduler = get_sheduler_adapter(scheduler, optimizer, *args, **kwargs)

    def step(self):
        return self.scheduler.step()
