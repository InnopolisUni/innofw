"""
This package is part of our framework's CORE, which is meant to give flexible support for metrics from different
libraries and frameworks via common abstract wrapper, currently it has support for:
- pytorch
- segmentation_models_pytorch
- sklearn


Metrics are used to monitor and measure the performance of a model (during training and testing).
"""

__all__ = ["get_metric_adapter", "Metric"]
#
import os
import importlib

#
import torch.nn as nn

#
from innofw.core.metrics.base import BaseMetricAdapter


def factory_method(name):
    return __METRIC_ADAP_DICT__[name]


__METRIC_ADAP_DICT__ = dict()


def get_metric_adapter(augs):
    suitable_adapters = [
        metric_adapter
        for metric_adapter in __METRIC_ADAP_DICT__.values()
        if metric_adapter.is_suitable_input(augs)
    ]
    if len(suitable_adapters) == 0:
        raise NotImplementedError()
    elif len(suitable_adapters):
        return suitable_adapters[0](augs)


def register_metrics_adapter(name):
    def register_function_fn(cls):
        if name in __METRIC_ADAP_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, BaseMetricAdapter):
            raise ValueError(
                "Class %s is not a subclass of %s" % (cls, BaseMetricAdapter)
            )
        __METRIC_ADAP_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[: file.find(".py")]
        module = importlib.import_module("innofw.core.metrics." + module_name)


class Metric(nn.Module):
    """
    Class for working with different metrics
    ...

    Attributes
    ----------

    Methods
    -------
    forward(x):
        function to perform metric
    """

    def __init__(self, metric):
        super().__init__()
        self.metric = get_metric_adapter(metric)

    def forward(self, *args, **kwargs):
        return self.metric(*args, **kwargs)

    def __repr__(self):
        return str(self.metric)
