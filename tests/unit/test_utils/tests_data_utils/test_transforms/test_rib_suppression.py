import cv2
import numpy as np
import pytest
from pathlib import Path

from innofw.utils.data_utils.transforms.rib_suppression import RibSuppression


@pytest.fixture
def sample_image():
    image_path = Path(__file__).parent.absolute() / "test_data/sample_image.png"
    return cv2.imread(str(image_path))


@pytest.fixture
def suppressed_image():
    image_path = Path(__file__).parent.absolute() / "test_data/suppressed_image.png"
    return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)


class TestRibSuppression:
    def test_valid_input(self, sample_image, suppressed_image):
        suppress = RibSuppression()
        result = suppress(sample_image)["image"]
        assert np.array_equal(result, suppressed_image)

    def test_non_numpy_input(self):
        suppress = RibSuppression()
        with pytest.raises(TypeError):
            suppress("invalid input")

    def test_unequal_shape_input(self, sample_image):
        suppress = RibSuppression()
        input_image = sample_image[:100, :100]
        result = suppress(input_image)["image"]
        assert result.shape == input_image.shape

    def test_invalid_model_path(self, sample_image, monkeypatch):
        suppress = RibSuppression()
        model_path = Path(__file__).parent.absolute() / "test_data/invalid_model.pt"
        monkeypatch.setattr(suppress, "model_suppression", model_path)
        with pytest.raises(RuntimeError):
            suppress(sample_image)["image"]

    def test_equalize_output(self, sample_image):
        suppress = RibSuppression(equalize_out=True)
        result = suppress(sample_image)["image"]
        assert isinstance(result, np.ndarray)

