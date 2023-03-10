#
import torch

from innofw.core.schedulers import register_scheduler_adapter
from innofw.core.schedulers.base import BaseSchedulerAdapter

#


@register_scheduler_adapter("torch_adapter")
class TorchAdapter(BaseSchedulerAdapter):
    """
    Class that adapts torch schedulers

    Methods
    -------
    is_suitable_input(schedulers)
        checks if the schedulers is from torch framework
    step()
        updates a learning rate
    """

    def __init__(self, scheduler, optimizer, *args, **kwargs):
        super().__init__(scheduler, optimizer, *args, **kwargs)

    @staticmethod
    def is_suitable_input(scheduler) -> bool:
        return issubclass(
            scheduler, torch.optim.lr_scheduler._LRScheduler
        )  # isinstance

    def step(self):
        self.scheduler.step()
