import logging
import os
import pathlib
import shutil

import cv2
import torch

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)
from innofw.core.datasets.image_infer import ImageFolderInferDataset
from innofw.core.datasets.segmentation import SegmentationDataset
from innofw.utils.data_utils.preprocessing.dicom_handler import dicom_to_img
from innofw.utils.data_utils.preprocessing.dicom_handler import img_to_dicom


class DirSegmentationLightningDataModule(BaseLightningDataModule):
    """
    A Class used for working with segmentations datasets in the following format:
        ├───dataset_name
                └── image
                |    | some_image_name1.png
                |    | ...
                |    | some_image_nameN.png
                |
                └── label
                    | some_image_name1.png
                    | ...
                    | some_image_nameN.png
    ...

    Attributes
    ----------
    channels_num : int
        number of channels in the image
    aug : dict
        The list of augmentations
    val_size: float
        The proportion of the dataset to include in the validation set

    Methods
    -------
    setup_train_test_val():
        setups and splits Datasets for train, test and validation

    """

    task = ["image-segmentation", "multiclass-image-segmentation"]
    framework = [Frameworks.torch]

    def __init__(
        self,
        train,
        test,
        infer=None,
        augmentations=None,
        channels_num: int = 3,
        val_size: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 1,
        random_seed: int = 42,
        stage=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            train=train,
            test=test,
            batch_size=batch_size,
            num_workers=num_workers,
            infer=infer,
            stage=stage,
            *args,
            **kwargs,
        )

        self.aug = augmentations
        self.channels_num = channels_num
        self.val_size = val_size
        self.random_seed = random_seed

    def setup_train_test_val(self, **kwargs):
        train_aug = self.get_aug(self.aug, "train")
        test_aug = self.get_aug(self.aug, "test")
        val_aug = self.get_aug(self.aug, "val")

        train_val = SegmentationDataset(
            os.path.join(self.train_source, "image"),
            os.path.join(self.train_source, "label"),
            train_aug,
        )
        val_size = int(len(train_val) * float(self.val_size))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train_val, [len(train_val) - val_size, val_size]
        )
        # Set validatoin augmentations for val
        setattr(self.val_dataset, "transform", val_aug)
        self.test_dataset = SegmentationDataset(
            os.path.join(self.test_source, "image"),
            os.path.join(self.test_source, "label"),
            test_aug,
        )

    def setup_infer(self):
        pass

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        pass


class DicomDirSegmentationLightningDataModule(
    DirSegmentationLightningDataModule
):
    dataset = ImageFolderInferDataset
    """
    A Class used for working with dicom segmentations datasets in the following format:
        ├───dataset_name
                └── image
                |    | some_image_name1.dcm
                |    | ...
                |    | some_image_nameN.dcm
                |
                └── label
                    | some_image_name1.png
                    | ...
                    | some_image_nameN.png
    ...

    Attributes
    ----------
    channels_num : int 
        number of channels in the image 
    aug : dict
        The list of augmentations
    val_size: float
        The proportion of the dataset to include in the validation set

    Methods
    -------
    setup_train_test_val():
        setups and splits Datasets for train, test and validation
    
    save_preds(preds, stage: Stages, dst_path: pathlib.Path):
        Saves inference predictions in Dicom format

    """

    def setup_train_test_val(self, **kwargs):
        train_aug = self.get_aug(self.aug, "train")
        self.test_aug = self.get_aug(self.aug, "test")
        val_aug = self.get_aug(self.aug, "val")

        dicom_train_path = os.path.join(self.train_source, "images")
        png_train_path = os.path.join(self.train_source, "png")
        shutil.rmtree(png_train_path, ignore_errors=True)
        os.makedirs(png_train_path)
        for dicom in os.listdir(dicom_train_path):
            dicom_to_img(
                os.path.join(dicom_train_path, dicom),
                os.path.join(png_train_path, dicom.replace("dcm", "png")),
            )

        dicom_test_path = os.path.join(self.test_source, "images")
        png_test_path = os.path.join(self.test_source, "png")

        shutil.rmtree(png_test_path, ignore_errors=True)
        os.makedirs(png_test_path)
        for dicom in os.listdir(dicom_test_path):
            dicom_to_img(
                os.path.join(dicom_test_path, dicom),
                os.path.join(png_test_path, dicom.replace("dcm", "png")),
            )

        train_val = SegmentationDataset(
            png_train_path,
            os.path.join(self.train_source, "labels"),
            train_aug,
        )
        val_size = int(len(train_val) * float(self.val_size))
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            train_val, [len(train_val) - val_size, val_size]
        )
        # Set validatoin augmentations for val
        setattr(self.val_dataset, "transform", val_aug)
        self.test_dataset = SegmentationDataset(
            png_test_path,
            os.path.join(self.test_source, "labels"),
            self.test_aug,
        )

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        dicoms = []
        sc_names = []
        shutil.rmtree(os.path.join(self.dicoms, "png"), ignore_errors=True)
        for i in os.listdir(self.dicoms):
            if i.endswith(".dcm"):
                dicoms.append(os.path.join(self.dicoms, i))
                sc_names.append("SC" + i)
        pred = [p for pp in preds for p in pp]
        for i, m in enumerate(pred):
            mask = m.clone()
            mask = mask[0]
            mask[mask < 0.1] = 0
            mask[mask != 0] = 1
            img = dicom_to_img(dicoms[i])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img[mask != 0] = 255
            img_to_dicom(img, dicoms[i], os.path.join(dst_path, sc_names[i]))
        logging.info(f"Saved results to: {dst_path}")

    def setup_infer(self):
        if isinstance(self.predict_dataset, self.dataset):
            return self.predict_dataset
        self.dicoms = str(self.predict_dataset)
        png_path = os.path.join(self.dicoms, "png")
        if not os.path.exists(png_path):
            os.makedirs(png_path)
        dicoms = [f for f in os.listdir(self.dicoms) if "dcm" in f]
        for dicom in dicoms:
            dicom_to_img(
                os.path.join(self.dicoms, dicom),
                os.path.join(png_path, dicom.replace("dcm", "png")),
            )
        self.predict_dataset = self.dataset(png_path, self.test_aug, True)
