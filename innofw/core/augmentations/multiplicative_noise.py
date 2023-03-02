from albumentations.augmentations import functional as F 
import albumentations as albu
import numpy as np


class MultiplicativeNoiseSelective(albu.MultiplicativeNoise):
    """Multiply image to random number or array of numbers.
    Args:
        multiplier (float or tuple of floats): If single float image will be multiplied to this number.
            If tuple of float multiplier will be in range `[multiplier[0], multiplier[1])`. Default: (0.9, 1.1).
        per_channel (bool): If `False`, same values for all channels will be used.
            If `True` use sample values for each channels. Default False.
        elementwise (bool): If `False` multiply multiply all pixels in an image with a random value sampled once.
            If `True` Multiply image pixels with values that are pixelwise randomly sampled. Defaule: False.
    Targets:
        image
    Image types:
        Any
    """

    def apply(self, img, multiplier=np.array([1]), **kwargs):
        c = img.shape[0]
        if np.random.random()<0.5:
            img[:int(c/2), ...] = F.multiply(img[:int(c/2), ...], multiplier)
            return img
        else:
            img[int(c/2):, ...] = F.multiply(img[int(c/2):, ...], multiplier)
            return img