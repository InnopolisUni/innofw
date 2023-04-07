import numpy as np
from collections import defaultdict
from numpy.testing import assert_array_equal
from numpy import max as npmax
from numpy import min as npmin
from numpy import zeros
from innofw.utils.data_utils.transforms.db_scan import *


def test_make_hist():
    img = np.array([[0, 0], [0, 0]])
    hist = make_hist(img)
    assert hist == defaultdict(int, {0: 4})

    hist = make_hist(np.array([[0, 1], [1, 0]]))
    assert hist == defaultdict(int, {0: 2, 1: 2})


def test_make_kernel_trick():
    kernel = make_kernel_trick(np.array([[0, 0], [0, 0]]))
    assert kernel == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]

    kernel = make_kernel_trick(np.array([[1, 2], [3, 4]]))
    assert kernel == [(1, 0, 0), (2, 0, 1), (3, 1, 0), (4, 1, 1)]

    kernel = make_kernel_trick(np.array([[4, 4], [4, 4]]))
    assert kernel == [(4, 0, 0), (4, 0, 1), (4, 1, 0), (4, 1, 1)]


def test_dekernel():
    zipped = [((1, 1, 1), 1), ((1, 1, 1), 2)]
    dekerneled = dekernel(zipped, shape=(2, 2))
    assert_array_equal(dekerneled, np.array([[0, 0], [0, 2]]))


def test_make_mask():
    img = np.array([[1, 0], [0, 1]])
    mask = make_mask(img, cluster=1)
    assert_array_equal(mask, np.array([[1, 0], [0, 1]]))

    mask = make_mask(img, cluster=2)
    assert_array_equal(mask, np.array([[0, 0], [0, 0]]))


def test_make_contrasted():
    contrasted = make_contrasted(np.array([[127, 127], [127, 127]]), contrast=50)
    assert_array_equal(contrasted, np.array([[127, 127], [127, 127]]))
