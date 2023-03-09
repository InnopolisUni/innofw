import cv2
import pytest

from innofw.utils.data_utils.transforms.db_scan import MakeContrasted
from innofw.utils.data_utils.transforms.rib_suppression import RibSuppression


@pytest.fixture
def rib_path():
    path = "tests/data/images/images_for_sh/medicine/00000005_003_jpg.rf.f694233bc76a3c7a12633fafd55e278d.jpg"
    return path


@pytest.fixture
def brain_path():
    path = "tests/data/images/images_for_sh/medicine/1.jpg"
    return path


def test_rib_suppression(rib_path):
    img = cv2.imread(rib_path)
    suppress = RibSuppression()
    suppressed = suppress(img)["image"]
    assert suppressed is not None


def test_db_scan(brain_path):
    img = cv2.imread(brain_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contraster = MakeContrasted()
    contrasted = contraster(img)["image"]
    assert contrasted is not None
