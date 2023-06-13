import torch
from abc import ABC, abstractmethod

class ImageToText(torch.nn.Module, ABC):
    """Base class for image to text architectures."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, image: torch.Tensor, captions: torch.Tensor=None, forcing=False) -> torch.Tensor:
        """Transforms the input image to a sequence of words.

        Args:
            x (torch.Tensor): Input image.
            captions (torch.Tensor, optional): Captions to use as input.
            forcing (bool, optional): Whether to use teacher forcing.

        Returns:
            torch.Tensor: Sequence of words.

        Shape:
            - x: :math:`(N, C, H, W)`
            - Output: :math:`(N, seq_len, vocab_size)`
        """
        pass