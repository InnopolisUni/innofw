from albumentations.core.transforms_interface import ImageOnlyTransform


class DivideBy255(ImageOnlyTransform):
    """
    sentinel2 preprocessing: dividing by 1e4 and clipping in range [0, 1]
    """

    def apply(self, img, **kwargs):
        return default_preprocessing(img)

    def get_transform_init_args_names(self):
        return ()


def default_preprocessing(img, divisor=255):
    img_copy = img.copy()
    img_copy = img_copy / divisor
    img_copy[img_copy > 1] = 1
    return img_copy



class StandardizeNDVI(ImageOnlyTransform):
    """
        ndvi to make images from -1 to 1 -> 0 to 1
    """

    def apply(self, img, **kwargs):
        return standardize_ndvi(img)

    def get_transform_init_args_names(self):
        return ()
    

def standardize_ndvi(img):
    """
        from -1 to 1 -> 0 to 1
    """
    img = (img + 1 ) /2
    return img

class ToFloatWClip(ImageOnlyTransform):
    """
    dividing by max_value and clipping in range [0, 1]
    """

    def __init__(self, max_value, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.max_value = max_value

    def apply(self, img, **kwargs):
        return default_preprocessing(img, self.max_value)

    def get_transform_init_args_names(self):
        return ()
