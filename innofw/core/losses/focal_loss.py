import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class FocalLoss(_Loss):
    """
    Focal loss implementation from `https://gitlab.com/chemrar/Pubmed/-/blob/dev/pubmed/loss/focal_loss.py`
    """

    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weights = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        probs = F.softmax(input, dim=1)
        probs = probs[torch.arange(probs.shape[0]), target]
        losses = -torch.log(probs) * (1 - probs) ** self.gamma
        if self.weights is not None:
            for index in torch.unique(target):
                losses[target == index] *= self.weights[index]
        return torch.mean(losses)
