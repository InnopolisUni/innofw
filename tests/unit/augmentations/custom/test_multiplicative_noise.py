import numpy as np
from innofw.core.augmentations.multiplicative_noise import MultiplicativeNoiseSelective
def test_multiplicative_noise():
    
    mn = MultiplicativeNoiseSelective()

    image = np.ones((64, 64, 3))
    for i in range(10):
        res = mn.apply(image, multiplier=np.array([2]))
        assert res is not None
        assert np.max(res) == 2 
