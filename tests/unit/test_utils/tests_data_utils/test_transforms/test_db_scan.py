import cv2
import numpy as np
from pathlib import Path
from innofw.utils.data_utils.transforms.db_scan import *
import pytest


@pytest.fixture
def img():
    return cv2.imread("path/to/image.png")


def test_norming(img):
    normalized = norming(img)
    assert (normalized >= 0).all()
    assert (normalized <= 1).all()


def test_make_hist(img):
    hist = make_hist(img)
    assert isinstance(hist, defaultdict)


def test_make_kernel_trick(img):
    kernel = make_kernel_trick(img)
    assert isinstance(kernel, list)


def test_dekernel():
    zipped = [((1, 2), 5), ((2, 3), 7)]
    img = dekernel(zipped, shape=(3, 4))
    assert isinstance(img, zeros)


def test_make_mask(img):
    mask = make_mask(img)
    assert isinstance(mask, zeros)


def test_make_contrasted(img):
    contrasted = make_contrasted(img)
    assert isinstance(contrasted, zeros)


class TestMakeContrasted:
    def test_call(self, img):
        contraster = MakeContrasted()
        output = contraster(img)
        assert isinstance(output, dict)
        assert "image" in output.keys()
        assert "bbox" in output.keys()
        assert "mask" in output.keys()
        assert "keypoints" in output.keys()
