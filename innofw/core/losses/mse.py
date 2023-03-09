from torch import functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


class MSELoss(_Loss):
    """
    A class to represent a MSE Loss that uses implementation of torch mse_loss function.
    MSELoss https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#MSELoss

    ...

    Attributes
    ----------
    size_average : Optional[bool]
    reduce : Optional[bool]
        Using  size_average and reduce is deprecated, it have same effect as using reduction:
            if size_average and reduce:
                ret = 'mean'
            elif reduce:
                ret = 'sum'
            else:
                ret = 'none'

    reduction: str


    Methods
    -------
     forward(self, input: Tensor, target: Tensor) -> Tensor:
        computes loss function
    """

    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean"
    ) -> None:
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)
