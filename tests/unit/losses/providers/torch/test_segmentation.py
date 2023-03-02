# # thrid-party libraries
# import torch
# from torch.nn import BCELoss
# import pytest

# # local modules
# from innofw.core.losses import Loss
import pytest
import torch

from innofw.core.losses.dice_loss_old import DiceLoss
from innofw.core.losses.focal_loss_old import FocalLoss
from innofw.core.losses.surface_loss_old import SurfaceLoss


def test_dice_loss():
    dice_loss = DiceLoss(
        beta=1.2,
        gamma=0.3,
        mode='log',
        alpha=0.6600361663652804
    )

    assert dice_loss(torch.tensor([0, 0.4, 0.4]), torch.tensor([0, 0, 1])) == pytest.approx(torch.tensor([0.8521]), 1e-3)
    assert dice_loss(torch.tensor([0.9, 0.4, 0.4]), torch.tensor([1, 0, 1])) == pytest.approx(torch.tensor([0.7843]), 1e-3)


def test_surface_loss():
    surface_loss = SurfaceLoss(
        scheduler=lambda m, i: min(0.3, m + i * 0.025)
    )
    surface_loss(torch.tensor([0, 0.4, 0.4]), torch.tensor([0, 0, 1])) 
    # assert surface_loss(torch.tensor([0, 0.4, 0.4]), torch.tensor([0, 0, 1])) == pytest.approx(torch.tensor([0.8521]), 1e-3)
    # assert surface_loss(torch.tensor([0.9, 0.4, 0.4]), torch.tensor([1, 0, 1])) == pytest.approx(torch.tensor([0.7843]), 1e-3)


# @pytest.mark.parametrize(["loss"], [[BCELoss()]])
# def test_torch_loss(loss):
#     loss = Loss(loss)

#     assert loss

#     pred = torch.tensor([0.0, 1.0, 0.0])
#     target = torch.tensor([1.0, 1.0, 0.0])
#     loss(pred, target)
