import os

import numpy as np

from innofw.utils.data_utils.preprocessing.dicom_handler import dicom_to_img
from innofw.utils.data_utils.preprocessing.dicom_handler import img_to_dicom
from innofw.utils.data_utils.preprocessing.dicom_handler import raster_to_dicom
from tests.utils import get_test_folder_path


def test_dicom_handler():

    dicom_path = os.path.join(
        get_test_folder_path(), "data/images/other/dicoms/test.dcm"
    )
    img = dicom_to_img(dicom_path)
    assert isinstance(img, np.ndarray)
    dicom = img_to_dicom(img)
    assert dicom

def test_raster_to_dicom():
    rgb_img = np.random.rand(124, 124, 3)
    ds = raster_to_dicom(rgb_img.astype(np.uint8), dicom=None)
    assert ds