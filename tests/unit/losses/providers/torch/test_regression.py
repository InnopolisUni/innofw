# thrid-party libraries
import torch
from torch.nn import L1Loss
import pytest

# local modules
from innofw.core.losses import Loss

"""
try these losses
# https://github.com/tensorflow/models/blob/master/research/object_detection/core/losses.py
# https://github.com/JunMa11/SegLoss
# https://github.com/qubvel/segmentation_models.pytorch/
"""


@pytest.mark.parametrize(["loss"], [[L1Loss()]])
def test_torch_loss(loss):
    loss = Loss(loss)

    assert loss

    pred = torch.tensor([0.0, 1.0, 0.0])
    target = torch.tensor([1.0, 1.0, 0.0])
    loss(pred, target)
