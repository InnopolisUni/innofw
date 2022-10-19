#
from typing import Callable

#
from torch.optim import SGD
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    CyclicLR,
    ConstantLR,
)
import pytest

#

from innofw.core.optimizers import Optimizer
from innofw.core.schedulers import Scheduler
from tests.fixtures.models.torch.dummy_model import DummyTorchModel

model = DummyTorchModel()
optimizer = Optimizer(SGD(model.parameters(), lr=100))


@pytest.mark.parametrize(
    ["scheduler", "args"],
    [
        [LambdaLR, {"lr_lambda": lambda epoch: 0.65**epoch}],
        [CosineAnnealingLR, {"T_max": 10}],
        [CosineAnnealingWarmRestarts, {"T_0": 10}],
        [StepLR, {"step_size": 20}],
        [CyclicLR, {"base_lr": 1e-3, "max_lr": 1e-2}],
    ],
)
def test_torch_scheduler(scheduler: Callable, args):
    scheduler = Scheduler(scheduler, optimizer=optimizer, **args)
    assert scheduler
