import logging
import math

import torch
from torch.nn import MSELoss
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR

from innofw.core.losses import Loss
from innofw.core.optimizers import Optimizer
from innofw.core.schedulers import Scheduler

#
#


def test_training():
    # ref https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#examples
    model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Flatten(0, 1))
    optimizer_wrp = Optimizer(RMSprop(model.parameters(), lr=1e-3))
    scheduler_wrp = Scheduler(
        CosineAnnealingLR, optimizer=optimizer_wrp, T_max=10, eta_min=0
    )

    loss_wrp = Loss(MSELoss(reduction="sum"))
    # Create Tensors to hold input and outputs.
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    # Prepare the input tensor (x, x^2, x^3).
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    for t in range(2000):
        y_pred = model(xx)
        loss = loss_wrp(y_pred, y)
        if t % 100 == 99:
            logging.info(f"{t}, {loss.item()}")

        optimizer_wrp.zero_grad()
        loss.backward()
        optimizer_wrp.step()
        scheduler_wrp.step()
