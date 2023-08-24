import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class FocalLoss(_Loss):
    def __init__(
        self, gamma=2, smooth=0.1, eps=1e-5, balanced=True, alpha=None
    ):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps
        self.balanced = balanced

        assert alpha is None or (alpha >= 0 and alpha <= 1)
        self.alpha = alpha

    def _get_weights(self, target, distances=None):
        if self.balanced:
            weights = target + 0  # todo: why?
            w1 = (
                self.alpha
                if self.alpha is not None
                else 1 - torch.mean(target)
            )
            weights[target > 0] = w1
            weights[target == 0] = 1 - w1
        else:
            weights = 1

        if distances is not None:
            weights = weights + distances

        return weights

    def _get_loss_value(self, output, target, weights):
        output = torch.clamp(output, min=self.eps, max=1 - self.eps)

        output = torch.stack((1.0 - output, output), dim=1)  # (N, C=2, H*W)
        logprobs = torch.clamp(torch.log(output), min=-100)  # (N, C=2, H*W)
        loss = F.nll_loss(
            logprobs, target.long(), reduction="none"
        )  # (N, H*W)
        pt = torch.exp(-loss)  # (N, H*W)

        # Label Smoothing
        smooth_loss = -logprobs.mean(dim=1)  # (N, H*W)
        loss = self.smooth * smooth_loss + (1 - self.smooth) * loss

        if self.gamma == 0:
            return (weights * loss).mean(dim=1)
        else:
            return (weights * (1.0 - pt) ** self.gamma * loss).mean(dim=1)

    def forward(self, output, target):
        output = output.view(output.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        return self._get_loss_value(
            output, target, self._get_weights(target.float())
        ).mean()
