import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class KLD(_Loss):
    """
            A class to represent a KL Divergence Loss that uses implementation from torch.
            `https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence`
            ...

            Attributes
            ----------

            Methods
            -------
             forward(self, q: torch.Tensor, p: torch.Tensor):
                computes loss function
        """
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, p: torch.Tensor):
        kl = torch.distributions.kl_divergence(q, p)
        return kl.mean()
