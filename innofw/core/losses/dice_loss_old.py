import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class DiceLoss(_Loss):

    def __init__(self, beta=1.2, gamma=1, mode='log', alpha=None):
        super().__init__()
        self.gamma = gamma
        self.mode = mode
        self.beta = beta

        assert alpha is None or (alpha >= 0 and alpha <= 1)
        self.alpha = alpha

    def get_dice_score(self, prediction, target, per_image=False, eps=1e-7):
        batch_size = target.shape[0]
        if not per_image:
            batch_size = 1

        if prediction.size() != target.size():
            target1, target2 = target[:,0], target[:,1]
            t = target[:,2]
            target = t * target1 + (1 - t) * target2

        prediction = prediction.view(batch_size, -1)
        target = target.reshape(batch_size, -1)

        tp = torch.sum(prediction * target, dim=1)
        fp = torch.sum(prediction, dim=1) - tp
        fn = torch.sum(target, dim=1) - tp

        fscores = ((1 + self.beta ** 2) * tp + eps) / \
                  ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + eps)

        return fscores

    def forward(self, output, target):
        output = F.logsigmoid(output).exp()
        d1 = self.get_dice_score(output, target)
        d0 = self.get_dice_score(1 - output, 1 - target)

        w1 = self.alpha if self.alpha is not None else 1 - torch.mean(target)
        w0 = 1 - w1

        if self.mode == 'linear':
            l1 = 1 - d1
            l0 = 1 - d0
        elif self.mode == 'log':
            l1 = (-torch.clamp(torch.log(torch.clamp(d1, min=1e-7, max=1)), min=-100)) ** self.gamma
            l0 = (-torch.clamp(torch.log(torch.clamp(d0, min=1e-7, max=1)), min=-100)) ** self.gamma
        else:
            raise Exception(f'DiceLoss: unsupported mode "{self.mode}" - "linear" or "log" allowed')
        # print('LOSSSS: ', (w1 * l1 + w0 * l0))
        return (w1 * l1 + w0 * l0)