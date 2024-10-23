# standard libraries
from typing import List, Optional, Generator, Tuple
import json
import os
import pathlib

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from innofw.constants import Frameworks, Stages
from innofw.core.datamodules.lightning_datamodules.base import BaseLightningDataModule


class FlorenceImageDataset(Dataset):
    def __init__(self, data_path=None, transform=None):
        if data_path.endswith("images"):
            data_path = data_path[:-7]
        self.data_path = data_path
        self.image_folder = os.path.join(data_path, "images")
        self.transform = transform
        self.data, self.size = self.setup()

    def setup(self):
        picture_formats = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        files = sorted(os.listdir(self.image_folder))
        data = [x for x in files if x.lower().endswith(picture_formats)]
        return data, len(data)

    def get_sample_name_text_input(self, item) -> Tuple[str, str]:
        text_input = None
        image_name = self.data[item]
        return image_name, text_input

    def __getitem__(self, item):
        image_name, text_input = self.get_sample_name_text_input(item)
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
        return self.size


class FlorenceJSONLDataset(FlorenceImageDataset):
    def setup(self):
        jsonl_path = os.path.join(self.data_path, "annotations.jsonl")
        return self.generate_json(jsonl_path), self.count_lines(jsonl_path)

    def get_sample_name_text_input(self, item) -> Tuple[str, str]:

        try:
            entry = self.data[item]
        except:
            entry = next(self.data)
        text_input = entry["prefix"].split("CAPTION_TO_PHRASE_GROUNDING ")[1]
        image_name = entry["image"]
        return image_name, text_input

    @staticmethod
    def read_data(jsonl_path, num_lines=None) -> List:
        """read data as List

        Args:
            num_lines: in case we need to limit size
            jsonl_path:

        Returns:

        """
        data = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if num_lines and i >= num_lines:
                    break
                data.append(json.loads(line.strip()))
        return data

    @staticmethod
    def count_lines(jsonl_path) -> int:
        valid_json_lines = 0
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        valid_json_lines += 1
                    except json.JSONDecodeError:
                        pass
        return valid_json_lines

    @staticmethod
    def generate_json(jsonl_path) -> Generator:
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    yield json.loads(line)


class FlorenceDataModuleAdapter(BaseLightningDataModule):
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

    dataset = FlorenceImageDataset
    # dataset = FlorenceJSONLDataset
    task = ["image-detection"]
    framework = [Frameworks.florence]

    def __init__(
        self,
        train: Optional[str],
        test: Optional[str],
        infer: Optional[str],
        batch_size: int = 4,
        augmentations=None,
        stage=Stages.predict,
        size_to=768,
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

        self.augmentations = augmentations
        self.infer = str(self.infer)
        self.stage = stage
        self.transform = None
        self.size_to = size_to

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
        new_size = (self.size_to, self.size_to)

        self.transform = transforms.Compose(
            [
                transforms.Resize(new_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict_dataloader(self):
        return self.dataset(self.infer, transform=self.transform)

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        basename = os.path.splitext(preds["file_name"])[0]
        prefix = "_" + preds["text_input"] if preds["text_input"] else ""
        file_name = basename + prefix + ".jpg"
        preds["image"].save(dst_path / file_name)

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass
