from albumentations.core.transforms_interface import ImageOnlyTransform


class DivideBy255(ImageOnlyTransform):
    """
    sentinel2 preprocessing: dividing by 1e4 and clipping in range [0, 1]
    """

    def apply(self, img, **kwargs):
        return default_preprocessing(img)

    def get_transform_init_args_names(self):
        return ()


def default_preprocessing(img):
    img_copy = img.copy()
    img_copy = img_copy / 255
    img_copy[img_copy > 1] = 1
    return img_copy
