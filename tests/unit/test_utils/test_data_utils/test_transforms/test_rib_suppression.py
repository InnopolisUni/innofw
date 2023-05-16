import cv2
import numpy as np
import torch
from pathlib import Path
from innofw.utils.data_utils.transforms.rib_suppression import *


def test_load_bone_model_pytorch():
    model = load_bone_model_pytorch()
    assert isinstance(model, torch.jit.ScriptModule)


def test_get_suppressed_image():
    img = np.zeros((512, 512), dtype=np.uint8)
    model = load_bone_model_pytorch()
    suppressed = get_suppressed_image(img, model, equalize_out=True)
    assert isinstance(suppressed, np.ndarray)
    assert suppressed.shape == img.shape


def test_RibSuppression_init():
    suppress = RibSuppression(equalize_out=True)
    assert suppress.equalize_out is True


def test_RibSuppression_call():
    suppress = RibSuppression()
    img = np.zeros((512, 512), dtype=np.uint8)
    output = suppress(image=img)
    assert isinstance(output, dict)
    assert isinstance(output['image'], np.ndarray)
    assert output['image'].shape == img.shape


def test_RibSuppression_call_with_color_image():
    suppress = RibSuppression()
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    output = suppress(image=img)
    assert isinstance(output, dict)
    assert isinstance(output['image'], np.ndarray)
    assert output['image'].shape == (512, 512)


def test_RibSuppression_call_with_invalid_image():
    suppress = RibSuppression()
    invalid_img = "invalid_image"
    try:
        suppress(image=invalid_img)
    except TypeError as e:
        assert "Expected type 'numpy.ndarray'" in str(e)
    else:
        assert False, "Expected TypeError for invalid image input"


def test_RibSuppression_call_with_bbox():
    suppress = RibSuppression()
    img = np.zeros((512, 512), dtype=np.uint8)
    bbox = [100, 100, 200, 200]
    output = suppress(image=img, bbox=bbox)
    assert isinstance(output, dict)
    assert isinstance(output['image'], np.ndarray)
    assert output['image'].shape == img.shape
    assert output['bbox'] == bbox


def test_RibSuppression_call_with_mask():
    suppress = RibSuppression()
    img = np.zeros((512, 512), dtype=np.uint8)
    mask = np.ones_like(img)
    output = suppress(image=img, mask=mask)
    assert isinstance(output, dict)
    assert isinstance(output['image'], np.ndarray)
    assert output['image'].shape == img.shape
    assert np.array_equal(output['mask'], mask)


def test_RibSuppression_call_with_keypoints():
    suppress = RibSuppression()
    img = np.zeros((512, 512), dtype=np.uint8)
    keypoints = [(100, 100), (200, 200)]
    output = suppress(image=img, keypoints=keypoints)
    assert isinstance(output, dict)
    assert isinstance(output['image'], np.ndarray)
    assert output['image'].shape == img.shape
    assert output['keypoints'] == keypoints


def test_RibSuppression_call_with_args_and_kwargs():
    suppress = RibSuppression()
    img = np.zeros((512, 512), dtype=np.uint8)
    try:
        suppress(image=img, arg1=1, arg2=2, kwarg1="kwarg1", kwarg2="kwarg2")
    except TypeError as e:
        assert "got an unexpected keyword argument" in str
