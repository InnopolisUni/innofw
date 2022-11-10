#
from typing import Optional

#
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRcnnModel(nn.Module):
    """
        FasterRcnn model for detection task
        ...

        Attributes
        ----------
        num_classes : int
            number of classes to predict
        model : nn.Module
            FasterRcnn model by torchvision

        Methods
        -------
        forward(x):
            returns result of the data forwarding

    """
    def __init__(
        self, num_classes: Optional[int] = 2, pretrained=True, *args, **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, t=None):
        return self.model(x, t)
