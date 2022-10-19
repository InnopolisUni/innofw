#
import torch.nn as nn
from torch.nn.modules.loss import _Loss

#
from innofw.core.losses import register_losses_adapter
from innofw.core.losses.base import BaseLossAdapter


@register_losses_adapter("torch_adapter")
class TorchAdapter(BaseLossAdapter):
    def __init__(self, loss, *args, **kwargs):
        super().__init__(loss)

    @staticmethod
    def is_suitable_input(loss) -> bool:
        return isinstance(loss, nn.modules.loss._Loss)

    def forward(self, *args, **kwargs):
        loss = self.loss(*args, **kwargs)
        return loss

    def __repr__(self):
        return f"Torch: {self.loss}"  # todo: serialize
