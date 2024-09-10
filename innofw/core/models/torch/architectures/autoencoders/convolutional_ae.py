import torch
import torch.nn as nn
from segmentation_models_pytorch import Unet


class CAE(nn.Module):
    def __init__(self, anomaly_threshold, input_channels=3):
        super(CAE, self).__init__()
        self.model = Unet(classes=input_channels, activation='sigmoid')
        self.anomaly_threshold = anomaly_threshold

    def forward(self, x):
        x_hat = self.model(x)
        return x_hat


if __name__ == '__main__':
    model = CAE(0)
    _x = torch.zeros((10, 3, 512, 512))
    print(model(_x).shape)