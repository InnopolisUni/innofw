import shutil
from pathlib import Path

import pytest

from innofw.utils.data_utils.dataset_prep.prep_yolov5 import prepare_dataset

import tempfile


class TestPrepareDataset:
    def test_prepare_dataset(self, data_dir, train_names_file, test_names_file, temp_dir):
        # Set up destination path
        dst_path = temp_dir / "output"
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
