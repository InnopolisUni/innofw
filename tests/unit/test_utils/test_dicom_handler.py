import os

import numpy as np

from innofw.utils.data_utils.preprocessing.dicom_handler import (
    dicom_to_img,
    img_to_dicom,
)
from tests.utils import get_test_folder_path


def test_dicom_handler():
    dicom_path = os.path.join(
        get_test_folder_path(), "data/images/other/dicoms/test.dcm"
    )
    img = dicom_to_img(dicom_path)
    assert isinstance(img, np.ndarray)
    dicom = img_to_dicom(img)
    assert dicom
