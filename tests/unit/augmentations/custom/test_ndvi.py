import numpy as np

from innofw.core.augmentations.preprocessing import StandardizeNDVI


def test_standardize_ndvi():
    preprocessing = StandardizeNDVI()

    img = np.random.randint(-1, 1, (2, 3, 224, 224))
    prep_img = preprocessing.apply(img)
    init_args = preprocessing.get_transform_init_args_names()

    assert init_args is ()

    assert prep_img.min() >= 0 and prep_img.max() <= 1
