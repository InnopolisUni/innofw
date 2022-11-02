from torch.nn.modules.loss import _Loss
from torch import functional as F, Tensor


# BCELoss https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#BCELoss
class BinaryCrossEntropyLoss(_Loss):
    def __init__(self, redaction="none", weight=None):
        super().__init__()
        self.reduction = redaction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(
            input, target, weight=self.weight, reduction=self.reduction
        )
