import torch
from torch.nn.modules.loss import _Loss


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _reduce(x, reduction):
    if reduction == "mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:
        return x


class IoUBatch(torch.nn.Module):
    __name__ = "iou_score"

    def __init__(
        self, eps=1e-7, threshold=0.5, per_image=True, reduction: str = "none"
    ):
        super().__init__()
        self.eps = eps
        self.threshold = threshold
        self.per_image = per_image
        self.reduction = reduction

    def forward(self, prediction, target):
        return self.run(prediction, target, self.threshold)

    def run(self, prediction, target, threshold):
        prediction = _threshold(prediction, threshold)

        batch_size = target.shape[0]
        if not self.per_image:
            batch_size = 1

        # deal with stacked mixup
        if prediction.size() != target.size():
            target1, target2 = target[:, 0], target[:, 1]
            t = target[:, 2]
            target = t * target1 + (1 - t) * target2

        prediction = prediction.view(batch_size, -1)
        target = target.reshape(batch_size, -1)

        intersection = torch.sum(prediction * target, dim=1)
        union = (
            torch.sum(prediction, dim=1)
            + torch.sum(target, dim=1)
            - intersection
        )
        iou_object = (intersection + self.eps) / (union + self.eps)

        return _reduce(iou_object, self.reduction)


class FScoreBatch(torch.nn.Module):
    __name__ = "f_score"

    def __init__(
        self,
        beta=1.0,
        eps=1e-7,
        threshold=0.5,
        per_image=True,
        reduction: str = "none",
    ):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.threshold = threshold
        self.per_image = per_image
        self.reduction = reduction

    def forward(self, prediction, target):
        return self.run(prediction, target, self.threshold)

    def run(self, prediction, target, threshold):
        prediction = _threshold(prediction, threshold)

        batch_size = target.shape[0]
        if not self.per_image:
            batch_size = 1

        # deal with stacked mixup
        if prediction.size() != target.size():
            target1, target2 = target[:, 0], target[:, 1]
            t = target[:, 2]
            target = t * target1 + (1 - t) * target2

        prediction = prediction.view(batch_size, -1)
        target = target.reshape(batch_size, -1)

        tp = torch.sum(prediction * target, dim=1)
        fp = torch.sum(prediction, dim=1) - tp
        fn = torch.sum(target, dim=1) - tp

        fscores = ((1 + self.beta**2) * tp + self.eps) / (
            (1 + self.beta**2) * tp + self.beta**2 * fn + fp + self.eps
        )

        return _reduce(fscores, self.reduction)


def get_metric_scores(prediction, target, funcs, threshold=0.5):
    scores = {}
    for key, metric_fn in funcs.items():
        scores[key] = metric_fn.run(prediction, target, threshold)
    return scores


class DiceLoss(_Loss):
    def __init__(self, beta=1.2, gamma: float = 1.0, mode="log", alpha=None):
        super().__init__()
        self.gamma = gamma
        self.mode = mode
        self.beta = beta
        self.fscore = FScoreBatch(
            beta=beta, threshold=None, per_image=False, reduction="none"
        )

        assert alpha is None or (alpha >= 0 and alpha <= 1)
        self.alpha = alpha

    def forward(self, output, target):
        d1 = self.fscore(output.squeeze(), target.squeeze())
        # default behaviour:
        #         d0 = torch.tensor([1])
        #         w1, w0 = 1, 0
        # new behaviour:
        d0 = self.fscore(1 - output, 1 - target)

        w1 = self.alpha if self.alpha is not None else 1 - torch.mean(target)
        w0 = 1 - w1

        if self.mode == "linear":
            l1 = 1 - d1
            l0 = 1 - d0
        elif self.mode == "log":
            l1 = (
                -torch.clamp(
                    torch.log(torch.clamp(d1, min=1e-7, max=1)), min=-100
                )
            ) ** self.gamma
            l0 = (
                -torch.clamp(
                    torch.log(torch.clamp(d0, min=1e-7, max=1)), min=-100
                )
            ) ** self.gamma
        else:
            raise Exception(
                f'DiceLoss: unsupported mode "{self.    mode}" - "linear" or "log" allowed'
            )

        return w1 * l1 + w0 * l0


# class DiceLoss(_Loss):

#     def __init__(self, beta=1.2, gamma: float=1.0, mode='log', alpha=None):
#         super().__init__()
#         self.gamma = gamma
#         self.mode = mode
#         self.beta = beta

#         assert alpha is None or (alpha >= 0 and alpha <= 1)
#         self.alpha = alpha

#     def get_dice_score(self, prediction, target, per_image=False, eps=1e-7):
#         batch_size = target.shape[0]
#         if not per_image:
#             batch_size = 1

#         if prediction.size() != target.size():
#             target1, target2 = target[:,0], target[:,1]
#             t = target[:,2]
#             target = t * target1 + (1 - t) * target2

#         prediction = prediction.view(batch_size, -1)
#         target = target.reshape(batch_size, -1)

#         tp = torch.sum(prediction * target, dim=1)
#         fp = torch.sum(prediction, dim=1) - tp
#         fn = torch.sum(target, dim=1) - tp

#         fscores = ((1 + self.beta ** 2) * tp + eps) / \
#                   ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + eps)

#         return fscores

#     def forward(self, output, target):
#         output = F.logsigmoid(output).exp()
#         d1 = self.get_dice_score(output, target)
#         d0 = self.get_dice_score(1 - output, 1 - target)

#         w1 = self.alpha if self.alpha is not None else 1 - torch.mean(target)
#         w0 = 1 - w1

#         if self.mode == 'linear':
#             l1 = 1 - d1
#             l0 = 1 - d0
#         elif self.mode == 'log':
#             l1 = (-torch.clamp(torch.log(torch.clamp(d1, min=1e-7, max=1)), min=-100)) ** self.gamma
#             l0 = (-torch.clamp(torch.log(torch.clamp(d0, min=1e-7, max=1)), min=-100)) ** self.gamma
#         else:
#             raise Exception(f'DiceLoss: unsupported mode "{self.mode}" - "linear" or "log" allowed')
#         # print('LOSSSS: ', (w1 * l1 + w0 * l0))
#         return (w1 * l1 + w0 * l0)
