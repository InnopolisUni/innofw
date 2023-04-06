import os.path
import tempfile
import pytest

from innofw.utils import get_project_root
from innofw.utils.data_utils.preprocessing.dicom_handler import *


@pytest.fixture(scope="module")
def test_data_dir():
    return os.path.join(get_project_root(), "path_to_test_data_dir")


@pytest.fixture(scope="module")
def test_image(test_data_dir):
    # Load test image from file and return as numpy array
    image_path = os.path.join(test_data_dir, "test_image.png")
    image = Image.open(image_path)
    return np.asarray(image)


@pytest.fixture(scope="module")
def test_dicom(test_data_dir):
    # Load test dicom from file and return as pydicom Dataset
    dicom_path = os.path.join(test_data_dir, "test_dicom.dcm")
    return dcmread(dicom_path)


@pytest.fixture(scope="module")
def tmp_dicom_file():
    # Create a temporary file to save DICOM for testing purposes
    with tempfile.NamedTemporaryFile(suffix=".dcm") as f:
        yield f.name


class TestDicomConversion:
    def test_img_to_dicom(self, test_image, tmp_dicom_file):
        # Convert image to DICOM and check the output
        dicom = img_to_dicom(test_image, path_to_save=tmp_dicom_file)
        assert isinstance(dicom, Dataset)
        assert os.path.exists(tmp_dicom_file)

    def test_dicom_to_img(self, test_dicom):
        # Convert DICOM to image and check the output
        img = dicom_to_img(test_dicom)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.dtype == np.uint8

    def test_raster_to_dicom(self, test_image):
        # Convert image to DICOM and check the output
        dicom = raster_to_dicom(test_image)
        assert isinstance(dicom, Dataset)

    def test_add_image(self, test_dicom, test_image):
        # Add image to DICOM and check the output
        updated_dicom = add_image(test_dicom, test_image)
        assert isinstance(updated_dicom, Dataset)
        assert updated_dicom.PixelData is not None
