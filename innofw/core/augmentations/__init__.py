"""
This package is part of our framework's CORE, which is meant to give flexible support for different augmentation
libraries and frameworks via common abstract wrapper, currently it has support for:
- albumentations
- torchvision

Augmentations are techniques used to increase the amount of data by adding slightly modified copies of already existing
data or newly created synthetic data from existing data. It acts as a regularizer and helps reduce overfitting when
training a machine learning model.
Augmentations can be used not only in the training stage, but in the testing as well. In that case it is called test time
augmentation.
"""

__all__ = ["get_augs_adapter", "Augmentation"]
#
import os
import importlib

#
import torch.nn as nn

#
from innofw.core.augmentations.base import BaseAugmentationAdapter


def factory_method(name):
    return __AUG_ADAP_DICT__[name]


__AUG_ADAP_DICT__ = dict()


def get_augs_adapter(augs):
    suitable_adapters = [
        aug_adapter
        for aug_adapter in __AUG_ADAP_DICT__.values()
        if aug_adapter.is_suitable_input(augs)
    ]
    if len(suitable_adapters) == 0:
        raise NotImplementedError("augmentation adapters are not found")
    elif len(suitable_adapters):
        return suitable_adapters[0](augs)


def register_augmentations_adapter(name):
    def register_function_fn(cls):
        if name in __AUG_ADAP_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, BaseAugmentationAdapter):
            raise ValueError(
                "Class %s is not a subclass of %s" % (cls, BaseAugmentationAdapter)
            )
        __AUG_ADAP_DICT__[name] = cls
        return cls

    return register_function_fn


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[: file.find(".py")]
        module = importlib.import_module("innofw.core.augmentations." + module_name)


class Augmentation(nn.Module):
    """
    Class provides same interface for different augmentations libraries

    Attributes
    ----------
    augs : adapter
        selected augmentations adapter
    Methods
    -------
    forward(x, y=None):
        perform augmentation
    """

    def __init__(self, augmentations):
        super().__init__()
        self.augs = None if augmentations is None else get_augs_adapter(augmentations)

    def forward(self, x, y=None):
        if self.augs is None:
            if y is not None:
                return x, y
            return x

        if y is not None:
            return self.augs(x, y)
        return self.augs(x)

    def __repr__(self):
        return str(self.augs)
