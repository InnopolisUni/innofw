from torch import functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


class BinaryCrossEntropyLoss(_Loss):
    """
    A class to represent a BCE Loss that uses implementation of torch binary_cross_entropy function.
    BCELoss https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#BCELoss

    ...

    Attributes
    ----------

    reduction: str
        reduction type to use: mean, sum, none
    weight: float
        a manual rescaling weight
    Methods
    -------
     forward(self, input: Tensor, target: Tensor) -> Tensor:
        computes loss function
    """

    def __init__(self, redaction="none", weight=None):
        super().__init__()
        self.reduction = redaction
        self.weight = weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(
            input, target, weight=self.weight, reduction=self.reduction
        )
