# This code was inspired by https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    def __init__(
        self, alpha=0.5, beta=0.5, gamma=1, weight=None, size_average=True
    ):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (
            TP + self.alpha * FP + self.beta * FN + smooth
        )
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky
