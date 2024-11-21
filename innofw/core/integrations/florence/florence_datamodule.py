# standard libraries
from typing import List, Optional, Generator, Tuple, Any
import json
import os
import pathlib

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from innofw.constants import Frameworks, Stages
from innofw.core.datamodules.lightning_datamodules.base import BaseLightningDataModule


class FlorenceDataset(Dataset):
    def __init__(self, data_path=None, transform=None):
        data_path = str(data_path)
        # when data is downloaded from S3 there is a bug
        if data_path.endswith("images"):
            data_path = data_path[:-7]  # care if this is deliberate
        self.data_path = data_path
        self.image_folder = os.path.join(data_path, "images")
        self.transform = transform
        self.data, self.len = self.setup()

    def setup(self):
        raise NotImplementedError

    def get_sample_name_text_input(self, item) -> Tuple[Any, str]:
        raise NotImplementedError

    def __getitem__(self, item):
        image_name, text_input = self.get_sample_name_text_input(item)
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        orig_size = (image.height, image.width)
        if self.transform:
            tensor = self.transform(image)
            tensor = tensor.unsqueeze(0)
        else:
            tensor = None
        return {
            "image": image,
            "image_path": image_path,
            "text_input": text_input,
            "image_name": image_name,
            "orig_size": orig_size,
            "tensor": tensor,
        }

    def __len__(self):
        return self.len


class FlorenceImageDataset(FlorenceDataset):
    """A dataset to represent image data for florence inference

    In case we use only pictures to retrieve results

    Expected folder structure:
    ---------------------------
    data_path/
    ├── images/              # A folder containing image files
    │   ├── image1.jpg       # Example image file
    │   ├── image2.png       # Example image file
    │   └── ...              # Additional image files

    """

    def setup(self) -> Tuple[Any, int]:
        picture_formats = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        files = sorted(os.listdir(self.image_folder))
        data = [x for x in files if x.lower().endswith(picture_formats)]
        return data, len(data)

    def get_sample_name_text_input(self, item):
        """since no text data text_input is None and should be provided via config"""
        text_input = None
        image_name = self.data[item]
        return image_name, text_input


class FlorenceJSONLDataset(FlorenceDataset):
    """JSONL dataset for florence

    Expected folder structure:
    ---------------------------
    data_path/
    ├── annotations.jsonl    # A JSONL file containing annotations
    ├── images/              # A folder containing image files
    │   ├── image1.jpg       # Example image file
    │   ├── image2.png       # Example image file
    │   └── ...              # Additional image files

    Notes:
    ------
    - The "annotations.jsonl" file should contain a valid JSONL format with one JSON object per line.
    - The "images" folder should contain the image files referenced in the annotations.
    """

    def setup(self):
        """setup dataset for jsonl data

        in order to work with List one may use read_data instead of generate_json

        Returns:

        """
        jsonl_path = os.path.join(self.data_path, "annotations.jsonl")
        return self.generate_json(jsonl_path), self.count_lines(jsonl_path)

    def get_sample_name_text_input(self, item):
        """parse data from jsonl records

        here .data can be both a generator or a List

        Args:
            item:

        Returns:

        """
        try:
            entry = self.data[item]
        except:
            entry = next(self.data)
        text_input = entry["prefix"].split("CAPTION_TO_PHRASE_GROUNDING ")[1]
        image_name = entry["image"]
        return image_name, text_input

    @staticmethod
    def read_data(jsonl_path, num_lines=None) -> List:
        """read jsonl data as List

        Args:
            num_lines: in case we need to limit size of read lines
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
        """Read number of lines in JSONL to get dataset size"""
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
        """Fast and easy way to produce samples via generator"""
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    yield json.loads(line)


class FlorenceImageDataModuleAdapter(BaseLightningDataModule):

    dataset = FlorenceImageDataset
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
        size_to=None,
        *args,
        **kwargs,
    ):
        """
        Arguments:
            train - path to the folder with train images
            test - path to the folder with test images
            batch_size -
            size_to - size images are to scale to
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


class FlorenceJSONLDataModuleAdapter(FlorenceImageDataModuleAdapter):
    dataset = FlorenceJSONLDataset
