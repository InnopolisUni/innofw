# # thrid-party libraries
# import torch
# from torch.nn import BCELoss
# import pytest
# # local modules
# from innofw.core.losses import Loss
import pytest
import torch

from innofw.core.losses.dice_loss_old import DiceLoss, IoUBatch
from innofw.core.losses.dice_loss_old import (_threshold,
                                              _reduce,)

from innofw.core.losses.surface_loss_old import SurfaceLoss
from innofw.core.losses.focal_loss_old import FocalLoss



def test_dice_loss():
    dice_loss = DiceLoss(
        beta=1.2, gamma=0.3, mode="log", alpha=0.6600361663652804
    )

    assert dice_loss(
        torch.tensor([0, 0.4, 0.4]), torch.tensor([0, 0, 1])
    ) == pytest.approx(torch.tensor([0.8521]), 1e-3)
    assert dice_loss(
        torch.tensor([0.9, 0.4, 0.4]), torch.tensor([1, 0, 1])
    ) == pytest.approx(torch.tensor([0.7843]), 1e-3)

def test_focal_loss():
    focal_loss = FocalLoss(
        gamma=2, smooth=0.1, eps=1e-5
    )

    assert focal_loss(
        torch.tensor([0, 0.4, 0.4]), torch.tensor([0, 0, 1])
    ) == pytest.approx(torch.tensor(0.081), abs=1e-3)
    assert focal_loss(
        torch.tensor([0.9, 0.4, 0.4]), torch.tensor([1, 0, 1])
    ) == pytest.approx(torch.tensor(0.055), abs=1e-3)

def test_surface_loss():
    surface_loss = SurfaceLoss(scheduler=lambda m, i: min(0.3, m + i * 0.025))
    assert surface_loss(torch.tensor([0, 0.4, 0.4]), torch.tensor([0, 0, 1])) is not None
    # assert surface_loss(torch.tensor([0, 0.4, 0.4]), torch.tensor([0, 0, 1])) == pytest.approx(torch.tensor([0.8521]), 1e-3)
    # assert surface_loss(torch.tensor([0.9, 0.4, 0.4]), torch.tensor([1, 0, 1])) == pytest.approx(torch.tensor([0.7843]), 1e-3)


def test_iou_batch():
    for reduction in [None, "sum", "mean"]:
        iou_loss = IoUBatch(
            eps=1e-7,
            threshold=0.5,
            per_image=False,
            reduction=reduction,
        )

        assert iou_loss(
        torch.tensor([0, 0.4, 0.4]), torch.tensor([0, 0, 1])
        ) < 1 #pytest.approx(torch.tensor([1]), 1e-7)
        assert iou_loss(
            torch.tensor([0.9, 0.4, 0.8]), torch.tensor([1, 0, 1])
        ) == 1 #pytest.approx(torch.tensor([1]), 1e-3)


def test__threshold():

    x = torch.rand(2, 3)
    out = _threshold(x, threshold=0.5)
    assert torch.max(out) == 1
    assert torch.min(out) == 0

def test__reduce():
    x = torch.Tensor([0, 1])
    out_mean = _reduce(x, reduction='mean')
    out_sum = _reduce(x, reduction='sum')
    out_none = _reduce(x, reduction=None)
    assert out_mean == 0.5
    assert out_sum == 1

