from torch.nn.modules.loss import _Loss
from torch import functional as F, Tensor

# MSELoss https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#MSELoss


class MSELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)
