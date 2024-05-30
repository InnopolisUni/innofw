# standard libraries
import pathlib
from pathlib import Path
from typing import List, Optional
import shutil
import glob

# third party libraries
from sklearn.model_selection import train_test_split

# local modules
from innofw.constants import Frameworks, Stages
from innofw.core.datamodules.base import BaseDataModule


class UltralyticsDataModuleAdapter(BaseDataModule):
    """Class defines adapter interface to conform to Ultralytics data specifications

    Attributes
    ----------
    task: List[str]
        the task the datamodule is intended to be used for
    framework: List[Union[str, Frameworks]]
        the model framework the datamodule is designed to work with


    Methods
    -------
    setup_train_test_val()
        creates necessary .yaml files for the Ultralytics package.
        splits training data into train and validation sets
        allocates files in folders
    setup_infer()
        creates necessary .yaml files for the Ultralytics package.
        allocates files in folders
    """

    task = ["image-detection"]
    framework = [Frameworks.ultralytics]

    def __init__(
            self,
            train: Optional[str],
            # val: Optional[str],  # todo: add this
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
            random_state: int = 42,
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
            # # In this datamodule, the train source should be the folder train itself not the folder "train/images"
            if str(self.train_source).endswith("images") or str(self.train_source).endswith(
                    "labels"):
                self.train_source = Path(str(self.train_source)[:-7])
        if self.test:
            self.test_source = Path(self.test)
            if str(self.test_source).endswith("images") or str(self.test_source).endswith("labels"):
                self.test_source = Path(str(self.test_source)[:-7])

        if self.infer:
            self.infer_source = (
                Path(self.infer)
                if not (type(self.infer) == str and self.infer.startswith("rts"))
                else self.infer
            )
            if str(self.infer_source).endswith("images") or str(self.infer_source).endswith(
                    "labels"):
                self.infer_source = Path(str(self.infer_source)[:-7])

        self.batch_size = batch_size
        self.imgsz: int = image_size
        self.workers: int = num_workers
        self.val_size = val_size
        self.num_classes = num_classes
        self.names = names
        self.random_state = random_state
        self.augmentations = augmentations
        self.is_keypoint = kwargs.get("is_keypoint", False)

    def setup_train_test_val(self, **kwargs):
        """
        Input folder structure is as follows:
        images/
            train/
            validation/
            test/

        labels/
            train/
            validation/
            test/


        Method will divide train folder's contents into train and val folders
        """
        # root_dir
        # self.train_source -> folder with images
        self.train_source = (
            self.train_source
            if self.train_source.name == "train"
            else self.train_source / "train"
        )
        self.test_source = (
            self.test_source
            if self.test_source.name == "test"
            else self.test_source / "test"
        )

        root_path = self.train_source.parent.parent

        # new data folder
        new_data_path = root_path / "innofw_split_data"
        new_data_path.mkdir(exist_ok=True, parents=True)

        new_img_path = new_data_path / "images"
        new_lbl_path = new_data_path / "labels"

        # === split train images and labels into train and val sets and move files ===

        # split images and labels
        train_img_path = self.train_source / "images" / "train"
        train_lbl_path = self.train_source / "labels" / "train"

        # get all files from train folder
        img_files = list(train_img_path.iterdir())
        label_files = list(train_lbl_path.iterdir())
        assert (
                len(label_files) == len(img_files) != 0
        ), "number of images and labels should be the same"

        # sort the files so that the images and labels are in the same order
        img_files.sort()
        label_files.sort()

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

        # Creating the images directory
        for files, folder_name in zip(
                [train_img_files, val_img_files], ["train", "val"]
        ):
            # create a folder
            new_path = new_img_path / folder_name
            new_path.mkdir(exist_ok=True, parents=True)

            # Copy files into folder
            for file in files:
                shutil.copy(file, new_path / file.name)

        # Creating the labels directory
        for files, folder_name in zip(
                [train_label_files, val_label_files], ["train", "val"]
        ):
            # create a folder
            new_path = new_lbl_path / folder_name
            new_path.mkdir(exist_ok=True, parents=True)

            # Copy files into folder
            for file in files:
                shutil.copy(file, new_path / file.name)

        self.data = str(root_path / "data.yaml")
        self.train_dataset = str(new_img_path / "train")
        self.val_dataset = str(new_img_path / "val")
        self.test_dataset = self.test_source

        # create a yaml file
        with open(self.data, "w+") as file:
            file.write(f"train: {self.train_dataset}\n")
            file.write(f"val: {self.val_dataset}\n")
            file.write(f"test: {self.test_dataset}\n")

            if self.is_keypoint:
                file.write(
                    "kpt_shape: [17, 3]\nflip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n")
            file.write(f"nc: {self.num_classes}\n")
            file.write(f"names: {self.names}\n")

    def setup_infer(self):
        if (
                type(self.infer_source) == str
                and self.infer_source.startswith("rts")
                or Path(self.infer_source).is_file()
        ):
            return
        # root_dir
        self.infer_source = Path(self.infer_source)
        # if self.infer_file:
        #     self.infer_source = (
        #         self.infer_source
        #         if self.infer_source.name == Path(self.infer_file).stem
        #         else self.infer_source / Path(self.infer_file).stem
        #     )
        for path in self.infer_source.rglob("*"):
            if path.is_file() and path.suffix not in [".txt", ".yaml", ".zip"]:
                self.infer_source = Path(path).parent
                break

        root_path = self.infer_source.parent

        self.data = str(root_path / "data.yaml")

        with open(self.data, "w+") as file:
            file.write(f"nc: {self.num_classes}\n")
            file.write(f"names: {self.names}\n")

    def predict_dataloader(self):
        pass

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        pass

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass
