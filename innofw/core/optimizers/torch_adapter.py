#
import torch

from innofw.core.optimizers import register_optimizers_adapter
from innofw.core.optimizers.base import BaseOptimizerAdapter

#


@register_optimizers_adapter("torch_adapter")
class TorchAdapter(BaseOptimizerAdapter):
    """
    Class that adapts torch optimizers

    Methods
    -------
    is_suitable_input(optimizer)
        checks if the optimizer is from torch framework
    step()
        updates a model parameters
    """

    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(optimizer)

    @staticmethod
    def is_suitable_input(optimizer) -> bool:
        return isinstance(optimizer, torch.optim.Optimizer)

    def step(self):
        self.optimizer.step()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def params(self):  # iterable
        return self.optimizer.params

    @property
    def defaults(self) -> dict:
        return self.optimizer.defaults
