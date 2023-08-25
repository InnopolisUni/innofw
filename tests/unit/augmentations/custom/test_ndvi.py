from innofw.core.augmentations.preprocessing import StandardizeNDVI

import numpy as np


def test_standardize_ndvi():
    preprocessing = StandardizeNDVI()

    img = np.random.randint(-1, 1, (2, 3, 224, 224))
    prep_img = preprocessing.apply(img)

    assert prep_img.min() >= 0 and prep_img.max() <= 1
