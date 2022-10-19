import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class KLD(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, p: torch.Tensor):
        kl = torch.distributions.kl_divergence(q, p)
        return kl.mean()
