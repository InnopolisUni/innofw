# standard libraries
import pathlib
import shutil
from pathlib import Path
from typing import List
from typing import Optional

from sklearn.model_selection import train_test_split

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.base import BaseDataModule

# third party libraries
# local modules


class YOLOV5DataModuleAdapter(BaseDataModule):
    """Class defines adapter interface to conform to YOLOv5 data specifications

    Attributes
    ----------
    task: List[str]
        the task the datamodule is intended to be used for
    framework: List[Union[str, Frameworks]]
        the model framework the datamodule is designed to work with


    Methods
    -------
    setup_train_test_val()
        creates necessary .yaml files for the YOLOv5 package.
        splits training data into train and validation sets
        allocates files in folders
    setup_infer()
        creates necessary .yaml files for the YOLOv5 package.
        allocates files in folders
    """

    task = ["image-detection"]
    framework = [Frameworks.torch]

    def predict_dataloader(self):
        pass

    def setup_infer(self):
        if type(self.infer_source) == str and self.infer_source.startswith(
            "rts"
        ) or Path(self.infer_source).is_file():
            return
        # root_dir
        self.infer_source = Path(self.infer_source)
        root_path = self.infer_source.parent.parent
        # new data folder
        new_data_path = root_path / "unarchived"
        new_data_path.mkdir(exist_ok=True, parents=True)

        new_img_path = new_data_path / "images"

        # === split train images and labels into train and val sets and move files ===

        # split images and labels
        infer_img_path = self.infer_source / "images"

        # get all files from train folder
        img_files = list(infer_img_path.iterdir())

        for files, folder_name in zip([img_files], ["infer"]):
            # create a folder
            new_path = new_img_path / folder_name
            new_path.mkdir(exist_ok=True, parents=True)

            # copy files into new folder
            for file in files:
                shutil.copy(file, new_path / file.name)

        self.data = str(new_data_path / "data.yaml")

        with open(self.data, "w+") as file:
            file.write(f"nc: {self.num_classes}\n")
            file.write(f"names: {self.names}\n")

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        pass

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def __init__(
        self,
        train: Optional[str],
        test: Optional[str],
        infer: Optional[str],
        num_workers: int,
        image_size: int,
        num_classes: int,
        names: List[str],
        batch_size: int = 4,
        val_size: float = 0.2,
        augmentations=None,
        stage=False,
        channels_num: int = 3,
        *args,
        **kwargs,
    ):
        """
        Arguments:
            train - path to the folder with train images
            test - path to the folder with test images
            batch_size -
            num_workers
            image_size - size of the input image
            num_classes - number of classes
            names -
            val_size - fraction size of the validation set
        """
        super().__init__(train, test, infer, stage=stage, *args, **kwargs)
        if self.train:
            self.train_source = Path(self.train)
        if self.test:
            self.test_source = Path(self.test)

        self.infer_source = (
            Path(self.infer)
            if not (type(self.infer) == str and self.infer.startswith("rts"))
            else self.infer
        )

        self.batch_size = batch_size
        # super().__init__(train, test, batch_size, num_workers)

        self.imgsz: int = image_size
        self.workers: int = num_workers
        self.val_size = val_size
        self.num_classes = num_classes
        self.names = names
        self.random_state = 42
        self.augmentations = augmentations

        # folder_name = self.train_dataset.stem

        # labels_path = Path(self.train_dataset).parent.parent / "labels"

    def setup_train_test_val(self, **kwargs):
        # root_dir
        root_path = self.train_source.parent.parent
        # new data folder
        new_data_path = root_path / "unarchived"
        new_data_path.mkdir(exist_ok=True, parents=True)

        new_img_path = new_data_path / "images"
        new_lbl_path = new_data_path / "labels"

        # === split train images and labels into train and val sets and move files ===

        # split images and labels
        train_img_path = self.train_source / "images"
        train_lbl_path = self.train_source / "labels"

        # get all files from train folder
        img_files = list(train_img_path.iterdir())
        label_files = list(train_lbl_path.iterdir())
        assert (
            len(label_files) == len(img_files) != 0
        ), "number of images and labels should be the same"

        # split into train and val
        (
            train_label_files,
            val_label_files,
            train_img_files,
            val_img_files,
        ) = train_test_split(
            label_files,
            img_files,
            test_size=self.val_size,
            random_state=self.random_state,
        )

        # get all files from test folder
        test_img_path = self.test_source / "images"
        test_lbl_path = self.test_source / "labels"

        test_img_files = list(test_img_path.iterdir())
        test_label_files = list(test_lbl_path.iterdir())

        assert len(test_img_files) == len(
            test_label_files
        ), "number of test images and labels should be the same"
        for files, folder_name in zip(
            [train_label_files, val_label_files, test_label_files],
            ["train", "val", "test"],
        ):
            # create a folder
            new_path = new_lbl_path / folder_name
            new_path.mkdir(exist_ok=True, parents=True)
            # copy files into folder
            for file in files:
                shutil.copy(file, new_path / file.name)

        for files, folder_name in zip(
            [train_img_files, val_img_files, test_img_files],
            ["train", "val", "test"],
        ):
            # create a folder
            new_path = new_img_path / folder_name
            new_path.mkdir(exist_ok=True, parents=True)

            # copy files into new folder
            for file in files:
                shutil.copy(file, new_path / file.name)

        self.data = str(new_data_path / "data.yaml")

        self.train_dataset = str(new_img_path / "train")
        self.val_dataset = str(new_img_path / "test")
        self.test_dataset = str(new_img_path / "test")
        # create a yaml file
        with open(self.data, "w+") as file:
            file.write(f"train: {self.train_source}\n")
            file.write(f"val: {self.val_dataset}\n")
            file.write(f"test: {self.test_dataset}\n")

            file.write(f"nc: {self.num_classes}\n")
            file.write(f"names: {self.names}\n")
