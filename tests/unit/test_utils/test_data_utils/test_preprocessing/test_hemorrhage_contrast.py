import os

import numpy as np
from pydicom import dcmread

from innofw.utils.data_utils.preprocessing.CT_hemorrhage_contrast import (resize, 
                                                                          normalize_minmax, 
                                                                          get_metadata_from_dicom,
                                                                          get_first_of_dicom_field_as_int, 
                                                                          prepare_image,
                                                                          window_image
)
from innofw.utils.data_utils.preprocessing.dicom_handler import dicom_to_img
from tests.utils import get_test_folder_path


def test_dicom_resize():
    dicom_path = os.path.join(
        get_test_folder_path(), "data/images/other/dicoms/test.dcm"
    )
    img = dicom_to_img(dicom_path)
    resize_img = resize(img, 256, 256)

    assert np.array(resize_img).shape[0] == 256
    assert np.array(resize_img).shape[1] == 256

def test_dicom_norm():
    dicom_path = os.path.join(
        get_test_folder_path(), "data/images/other/dicoms/test.dcm"
    )
    img = dicom_to_img(dicom_path)

    norm_img = normalize_minmax(img)

    assert np.max(np.array(norm_img)) <= 255
    assert np.min(np.array(norm_img)) >= 0

def test_dicom_metadata():
    dicom_path = os.path.join(
        get_test_folder_path(), "data/images/other/dicoms/test.dcm"
    )
    img = dcmread(dicom_path)

    # img_id, prep_img = prepare_image(img)
    metadata = get_metadata_from_dicom(img)
    window_center = metadata['window_center']
    window_width = metadata['window_width']
    slope = metadata['slope']
    assert isinstance(window_center, int)
    assert window_width == 200
    assert isinstance(slope, int)

def test_metadata_int():
    # dicom_path = os.path.join(
    #     get_test_folder_path(), "data/images/other/dicoms/test.dcm"
    # )
    metadata = {
        "window_center": 50.0,
        "window_width": float(200),
        "intercept": -(2049/2),
        "slope": 1.0,
    }
    new_metadata = {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

    for param in new_metadata.keys():
        assert isinstance(new_metadata[param], int)


def test_prepare_image():
    import PIL
    dicom_path = os.path.join(
        get_test_folder_path(), "data/images/other/dicoms/test.dcm"
    )
    img = dcmread(dicom_path)

    img_id, prep_img = prepare_image(img)
    # array_img = np.array(prep_img)
    assert isinstance(prep_img, PIL.Image.Image)
    assert isinstance(img_id, str)


def test_window_image():
    dicom_path = os.path.join(
        get_test_folder_path(), "data/images/other/dicoms/test.dcm"
    )
    img = dcmread(dicom_path)

    window_center = 50
    window_width = 200
    intercept = 0
    slope = 1

    out_img = window_image(img.pixel_array, window_center, window_width, intercept, slope)
    assert out_img is not None




    
