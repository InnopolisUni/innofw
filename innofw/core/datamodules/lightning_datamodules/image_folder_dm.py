import logging
import os.path
import pathlib
import rasterio

import pandas as pd
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)
from innofw.core.datasets.utils import stratified_split

#
#


class ImageLightningDataModule(BaseLightningDataModule):
    """Class defines dataset preparation and dataloader creation for image classification

    Attributes
    ----------
    task: List[str]
        the task the datamodule is intended to be used for
    framework: List[Union[str, Frameworks]]
        the model framework the datamodule is designed to work with

    Methods
    -------
    setup_train_test_val
        splits train data into train and validation sets
        creates dataset objects
    save_preds(preds, stage: Stages, dst_path: pathlib.Path)
        saves predicted class labels for images in a .csv file
    """

    task = ["image-classification"]
    framework = [Frameworks.torch]

    def __init__(
        self,
        train,
        test,
        batch_size: int = 16,
        val_size: float = 0.2,
        num_workers: int = 1,
        augmentations=None,
        infer=None,
        stage=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            train, test, infer, batch_size, num_workers, stage=stage
        )
        self.aug = augmentations
        self.val_size = val_size
        if "stratify" in kwargs.keys():
            self.stratify = kwargs["stratify"]
        else:
            self.stratify = False

    def setup_train_test_val(self, **kwargs):
        train_aug = self.get_aug(self.aug, "train")
        test_aug = self.get_aug(self.aug, "test")
        val_aug = self.get_aug(self.aug, "val")

        if os.listdir(os.path.join(self.train_source, os.listdir(self.train_source)[0]))[0].endswith(".tif"):
            train_dataset = ImageFolder(
                str(self.train_source), transform=train_aug, loader=lambda path: rasterio.open(path).read().transpose((1, 2, 0)),
            )
            self.test_dataset = ImageFolder(
                str(self.train_source), transform=test_aug, loader=lambda path: rasterio.open(path).read().transpose((1, 2, 0)),
            )
        else:
            train_dataset = ImageFolder(
            str(self.train_source), transform=train_aug)
        self.test_dataset = ImageFolder(
            str(self.train_source), transform=test_aug
        )
        # divide into train, val, test
        n = len(train_dataset)
        train_size = int(n * (1 - self.val_size))


        # stratify
        if self.stratify:
            first_set_inputs, first_set_labels, second_set_inputs, second_set_labels = stratified_split(train_dataset,
                                                                                                    train_dataset.targets,
                                                                                                    1 - self.val_size)
            from collections import Counter
            strat_train = Counter(first_set_labels).values()
            coefsstrat_train = [i/len(first_set_labels) for i in strat_train]

            strat_val = Counter(second_set_labels).values()
            coefsstrat_val = [i/len(second_set_labels) for i in strat_val]

            random_train = Counter([self.train_dataset.dataset.targets[i] for i in self.train_dataset.indices]).values()
            coefsrandom_train = [i/len(self.train_dataset.indices) for i in random_train]

            random_val = Counter([self.val_dataset.dataset.targets[i] for i in self.val_dataset.indices]).values()
            coefsrandom_val = [i/len(self.val_dataset.indices) for i in random_val]

            all_train = Counter(train_dataset.targets).values()
            coefsall_train = [i/len(train_dataset.targets) for i in all_train]

            self.train_dataset = first_set_inputs
            self.val_dataset = second_set_inputs

            print(f'''Coefficients of classes:\n 
            all dataset - {coefsall_train}\n
            stratified train split - {coefsstrat_train}\n 
            stratified val split - {coefsstrat_val}\n 
            random train split - {coefsrandom_train}\n 
            random val split - {coefsrandom_val}\n ''')
        else:
            self.train_dataset, self.val_dataset = random_split(
                train_dataset, [train_size, n - train_size])
        # Set validatoin augmentations for val
        setattr(self.val_dataset.dataset, "transform", val_aug)

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        out = []
        for sublist in preds:
            out.extend(sublist.tolist())
        images = self.predict_dataset.image_names
        df = pd.DataFrame(
            list(zip(images, out)), columns=["Image name", "Class"]
        )
        dst_filepath = os.path.join(dst_path, "classification.csv")
        df.to_csv(dst_filepath)
        logging.info(f"Saved results to: {dst_filepath}")
