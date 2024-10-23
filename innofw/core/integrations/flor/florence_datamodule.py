# standard libraries
from typing import List, Optional
import json
import os
import pathlib

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from innofw.constants import Frameworks, Stages
from innofw.core.datamodules.base import BaseDataModule


class FLorenceDataset(Dataset):
    def __init__(self, data, data_path=None, transform=None):
        self.data = data
        self.data_path = data_path
        self.image_folder = os.path.join(data_path, "images")
        self.image_to = (768, 768)
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(self.image_to),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

    def __getitem__(self, item):
        entry = self.data[item]
        image_name = entry["image"]
        prefix = entry["prefix"]
        # Извлекаем подсказку
        text_input = prefix.split("CAPTION_TO_PHRASE_GROUNDING ")[1]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        orig_size = (image.height, image.width)
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        return {
            "image": image,
            "image_path": image_path,
            "text_input": text_input,
            "image_name": image_name,
            "orig_size": orig_size,
            "tensor": tensor,
        }

    def __len__(self):
        return len(self.data)


class FlorDataModuleAdapter(BaseDataModule):
    """

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

    dataset = FLorenceDataset
    task = ["image-detection"]
    framework = [Frameworks.flor]

    def __init__(
        self,
        train: Optional[str],
        test: Optional[str],
        infer: Optional[str],
        batch_size: int = 4,
        augmentations=None,
        stage=False,
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

        self.batch_size = batch_size
        self.augmentations = augmentations
        self.infer = str(self.infer)

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
        pass

    def setup_infer(self):

        jsonl_path = os.path.join(self.infer, "annotations.jsonl")
        num_lines = 300
        self.data = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                self.data.append(json.loads(line.strip()))

    def predict_dataloader(self):
        return self.dataset(self.data, self.infer)

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        basename = os.path.splitext(preds["file_name"])[0]
        file_name = basename + "_" + preds["text_input"] + ".jpg"
        preds["image"].save(dst_path  / file_name)

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass
