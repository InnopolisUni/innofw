import torch
from torch.optim import Adam
from torch.optim import Optimizer as TorchOptim
from torch.optim import SGD as Sgd

from innofw.core.optimizers import Optimizer

#


class SGD(Optimizer):
    """
    Class defines a wrapper of the torch optimizer to illustrate
     how to use custom optimizer implementations in the innofw

     Attributes
     ----------
     optimizer
        optimizer from torch framework
    """

    def __init__(self, *args, **kwargs):
        super().__init__(optimizer=None)
        self.optim = Sgd(*args, **kwargs)


class ADAM(Optimizer):
    """
    Class defines a wrapper of the torch optimizer to illustrate
    how to use custom optimizer implementations in the innofw
    Attributes
    ----------
    optimizer : torch.optim
        optimizer from torch framework
    """

    def __init__(self, *args, **kwargs):
        super().__init__(optimizer=None)
        self.optim = Adam(*args, **kwargs)


class LION(Optimizer):  # [1]
    """
    [1] Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., ... & Le,
    Q. V. (2023). Symbolic Discovery of Optimization Algorithms. arXiv
    preprint arXiv:2302.06675.

    Implementation is taken from https://github.com/google/automl/tree/master
    """

    class _Lion(TorchOptim):
        r"""Implements Lion algorithm."""

        def __init__(self, params, lr=1e-4, b1=0.9, b2=0.99, wd=0.0):
            if not 0.0 <= lr:
                raise ValueError("Invalid learning rate: {}".format(lr))
            if not 0.0 <= b1 < 1.0:
                raise ValueError("Invalid beta parameter at index 0: {}".format(b1))
            if not 0.0 <= b2 < 1.0:
                raise ValueError("Invalid beta parameter at index 1: {}".format(b2))
            defaults = dict(lr=lr, b1=b1, b2=b2, wd=wd)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            """Performs a single optimization step.
            Args:
              closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            Returns:
              the loss.
            """
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    p.data.mul_(1 - group["lr"] * group["wd"])

                    grad = p.grad
                    state = self.state[p]
                    # State initialization
                    if len(state) == 0:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p)

                    exp_avg = state["exp_avg"]
                    beta1, beta2 = group["b1"], group["b2"]

                    # Weight update
                    update = exp_avg * beta1 + grad * (1 - beta1)
                    p.add_(torch.sign(update), alpha=-group["lr"])
                    # Decay the momentum running average coefficient
                    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

            return loss

    def __init__(self, *args, **kwargs):
        super().__init__(optimizer=None)
        self.optim = self._Lion(*args, **kwargs)
