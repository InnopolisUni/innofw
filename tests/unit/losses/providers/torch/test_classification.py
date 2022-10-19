# thrid-party libraries
import torch
from torch.nn import BCELoss
import pytest

# local modules
from innofw.core.losses import Loss


@pytest.mark.parametrize(["loss"], [[BCELoss()]])
def test_torch_loss(loss):
    loss = Loss(loss)

    assert loss

    pred = torch.tensor([0.0, 1.0, 0.0])
    target = torch.tensor([1.0, 1.0, 0.0])
    loss(pred, target)
