import hydra
import torch.nn as nn

class WeightInitializer:
    """
    Class used for working with data from s3 storage
    Attributes
    ----------
    init_func :
        weight initialization function
    layers :
        layers
    bias : float
        bias

    Methods
    -------
    init_weights(model):
        initializes model's weights
    """

    def __init__(
        self,
        init_func,
        layers=(nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.Linear),
        bias=0.01,
    ):
        self.init_func = init_func
        self.layers = layers
        self.bias = bias

    def init_weights(self, model):
        for m in model.modules():
            if any(isinstance(m, layer) for layer in self.layers):
                try:
                    hydra.utils.instantiate(self.init_func, tensor=m.weight)
                except:
                    pass

                if m.bias is not None:
                    try:
                        m.bias.data.fill_(self.bias)
                    except:
                        pass
