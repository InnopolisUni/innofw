import numpy as np
import torch
import multiprocessing
from scipy import ndimage
from torch.nn.modules.loss import _Loss

class SurfaceLoss(_Loss):

    def __init__(self, activation='sigmoid'):
        super().__init__()
        self.activation = activation

    def _compute_distance(self, target):
        return np.where(
            target > 0,
            -ndimage.distance_transform_edt(target),
            ndimage.distance_transform_edt(1 - target),
        )

    def forward(self, output, target):
        distances = torch.tensor(
            self._compute_distance(target.detach().cpu().numpy()),
            device=target.device,
        )

        distances = distances.view(distances.shape[0], -1)
        if self.activation == 'sigmoid':
            output = torch.sigmoid(output)
        else:
            output = torch.softmax(output)
        output = torch.sigmoid(output)
        output = output.view(output.shape[0], -1)
        loss = torch.sum(distances * output, dim=1) / output.shape[1]
        return  0.3 * 0.01 * loss.mean()
