from innofw.core.augmentations.preprocessing import DivideBy255, ToFloatWClip
import numpy as np

def test_divide_by_255():
    transf = DivideBy255()

    image = np.ones((64, 64, 3)) * 255
    res_image = transf.apply(image)
    init_args = transf.get_transform_init_args_names()

    assert init_args is ()
    assert np.min(res_image)==1 and np.max(res_image)==1

def test_to_float_w_clip():
    transf = ToFloatWClip(max_value=100)
    
    image = np.ones((64, 64, 3)) * 100
    res_image = transf.apply(image)
    init_args = transf.get_transform_init_args_names()

    assert init_args is ()
    assert np.min(res_image)==1 and np.max(res_image)==1