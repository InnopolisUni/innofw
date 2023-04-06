import shutil
import tempfile
from pathlib import Path

import pytest

from innofw.utils.data_utils.dataset_prep.prep_yolov5 import prepare_dataset


@pytest.fixture(scope="module")
def test_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        yield tmp_dir


@pytest.fixture(scope="module")
def data_dir(test_dir):
    data_dir = test_dir / "data"
    data_dir.mkdir()
    # create dummy image and label files
    for i in range(10):
        with open(data_dir / f"{i}.jpeg", "w") as f:
            f.write("image file")
        with open(data_dir / f"{i}.txt", "w") as f:
            f.write("label file")
    return data_dir


@pytest.fixture(scope="module")
def train_names_file(test_dir):
    train_names_file = test_dir / "train.txt"
    with open(train_names_file, "w") as f:
        for i in range(5):
            f.write(f"{i}\n")
    return train_names_file


@pytest.fixture(scope="module")
def test_names_file(test_dir):
    test_names_file = test_dir / "test.txt"
    with open(test_names_file, "w") as f:
        for i in range(5, 10):
            f.write(f"{i}\n")
    return test_names_file


class TestPrepareDataset:
    def test_prepare_dataset(self, data_dir, train_names_file, test_names_file, test_dir):
        # Set up destination path
        dst_path = test_dir / "output"
        dst_path.mkdir()

        # Test prepare_dataset function
        prepare_dataset(
            src_img_path=data_dir,
            src_label_path=data_dir,
            train_names_file=train_names_file,
            test_names_file=test_names_file,
            dst_path=dst_path,
        )

        # Check that train, test, and infer directories are created
        for d in ["train", "test", "infer"]:
            assert (dst_path / d).is_dir()
            assert (dst_path / d / "images").is_dir()
            assert (dst_path / d / "labels").is_dir()

        # Check that correct number of files were copied
        assert len(list((dst_path / "train" / "images").glob("*"))) == 5
        assert len(list((dst_path / "train" / "labels").glob("*"))) == 5
        assert len(list((dst_path / "test" / "images").glob("*"))) == 5
        assert len(list((dst_path / "test" / "labels").glob("*"))) == 5
        assert len(list((dst_path / "infer" / "images").glob("*"))) == 5
        assert len(list((dst_path / "infer" / "labels").glob("*"))) == 5
