import numpy as np
from collections import defaultdict
from numpy.testing import assert_array_equal
from numpy import max as npmax
from numpy import min as npmin
from numpy import zeros
from innofw.utils.data_utils.transforms.db_scan import *
import pytest


@pytest.mark.parametrize("img, expected_hist", [
    (np.array([[0, 0], [0, 0]]), defaultdict(int, {0: 4})),
    (np.array([[0, 1], [1, 0]]), defaultdict(int, {0: 2, 1: 2})),
    # Add more test cases as needed
])
def test_make_hist(img, expected_hist):
    hist = make_hist(img)
    assert hist == expected_hist


@pytest.mark.parametrize("input_array, expected_output", [
    (np.array([[0, 0], [0, 0]]), [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]),
    (np.array([[1, 2], [3, 4]]), [(1, 0, 0), (2, 0, 1), (3, 1, 0), (4, 1, 1)]),
    (np.array([[4, 4], [4, 4]]), [(4, 0, 0), (4, 0, 1), (4, 1, 0), (4, 1, 1)])
])
def test_make_kernel_trick(input_array, expected_output):
    kernel = make_kernel_trick(input_array)
    assert kernel == expected_output


def test_dekernel():
    zipped = [((1, 2, 3), 1), ((4, 5, 6), 2)]
    dekerneled = dekernel(zipped, shape=(2, 2))
    assert_array_equal(dekerneled, np.array([[0, 0], [0, 0]]))


@pytest.mark.parametrize("img, cluster, expected_output", [
    (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]), 1, np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])),
    (np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]), 0, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
])
def test_make_mask(img, cluster, expected_output):
    mask = make_mask(img, cluster=cluster)
    assert np.array_equal(mask, expected_output)


def test_make_contrasted():
    contrasted = make_contrasted(np.array([[0, 0], [127, 127]]), contrast=2)
    assert_array_equal(contrasted, np.array([[-4, -4], [127, 127]]))
