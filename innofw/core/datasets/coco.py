import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import pydicom
from gitdb.util import basename
from torch.utils.data import Dataset

from innofw.utils.data_utils.preprocessing.dicom_handler import (
    dicom_to_img,
    dicom_to_raster,
)


class CocoDataset(Dataset):
    """
    A class to represent a Coco format Dataset.
    https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html
    ...

    Attributes
    ----------
    dataframe : pandas.DataFrame
        df with train data info
    image_dir : str
        directory containing images
    transforms : Iterable[albumentations.augmentations.transforms]


    Methods
    -------
    __getitem__(self, index: int):
        returns image and target tensors and image_id

    read_image(self, path):
        reads an image using opencv
    """

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()
        if "image_id" not in dataframe:
            raise ValueError("image_id column should be in DataFrame")
        self.image_ids = dataframe["image_id"].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df["image_id"] == image_id]
        image = self.read_image(os.path.join(self.image_dir, str(image_id)))
        image = image / 255.0

        boxes = records[["x", "y", "w", "h"]].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x_max = x_min(x) + w
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y_max = y_min(y) + h

        area = records["box_area"].values
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            sample = {
                "image": image,
                "bboxes": target["boxes"],
                "labels": labels,
            }
            sample = self.transforms(**sample)
            image = sample["image"]

            target["boxes"] = torch.stack(
                tuple(map(torch.tensor, zip(*sample["bboxes"])))
            ).permute(1, 0)

        return image.float(), target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def read_image(self, path):
        image = cv2.imread(path + ".jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        return image


class DicomCocoDataset(CocoDataset):
    """
    A class to represent a Dicom Coco format Dataset.

    Methods
    -------
    read_image(self, path):
        reads an image using dicom handler, and converts dicom file to img array
    """

    def read_image(self, path):
        return dicom_to_img(path + ".dcm")


class DicomCocoDatasetInfer(Dataset):
    """
    A class to represent a Dicom Coco format Dataset for inference.

    dicom_dir : str
        directory containing images
    transforms : Iterable[albumentations.augmentations.transforms]

    Methods
    -------
    __getitem__(self, idx):
        return image
    """

    def __init__(self, dicom_dir, transforms=None):
        self.images = []
        self.paths = [os.path.join(dicom_dir, d) for d in os.listdir(dicom_dir)]
        for dicom in self.paths:
            self.images.append(transforms(image=dicom_to_img(dicom))["image"])

    def __getitem__(self, idx):
        return self.images[idx]

    def __len__(self):
        return len(self.images)


class WheatDataset(Dataset):
    """
    A dataset example for GWC 2021 competition. A class to represent a Coco format Dataset.
    https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html
    ...

    Attributes
    ----------
    annotations (string): Data frame with annotations.
    root_dir (string): Directory with all the images.
    transform (callable, optional): Optional data augmentation to be applied on a sample.


    Methods
    -------
    __getitem__(self, index: int):
        returns image, bboxes, domain
    """

    def __init__(self, annotations, root_dir, transforms=None):
        """
        Args:
            annotations (string): Data frame with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional data augmentation to be applied
                on a sample.
        """

        self.root_dir = Path(root_dir)
        self.image_list = annotations["image_name"].values
        self.domain_list = annotations["domain"].values
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        imgp = str(self.root_dir / (self.image_list[idx] + ".png"))
        domain = self.domain_list[
            idx
        ]  # We don't use the domain information but you could !
        bboxes = self.boxes[idx]
        img = cv2.imread(imgp)
        image = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # Opencv open images in BGR mode by default

        if self.transforms:
            transformed = self.transforms(
                image=image,
                bboxes=bboxes,
                class_labels=["wheat_head"] * len(bboxes),
            )  # Albumentations can transform images and boxes
            image = transformed["image"]
            bboxes = transformed["bboxes"]

        if len(bboxes) > 0:
            bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
            bboxes = torch.zeros((0, 4))
        return image, bboxes, domain

    def decodeString(self, BoxesString):
        """
        Small method to decode the BoxesString
        """
        if BoxesString == "no_box":
            return np.zeros((0, 4))
        else:
            try:
                boxes = np.array(
                    [
                        np.array([int(i) for i in box.split(" ")])
                        for box in BoxesString.split(";")
                    ]
                )
                return boxes
            except:
                print(BoxesString)
                print("Submission is not well formatted. empty boxes will be returned")
                return np.zeros((0, 4))


class DicomCocoDataset_rtk(Dataset):
    def __init__(self, *args, **kwargs):
        """
        Args:
            data_dir (str): Путь к директории с DICOM файлами и COCO аннотациями.
            transform (callable, optional): Трансформации, применяемые к изображениям и маскам.
        """
        data_dir = kwargs["data_dir"]
        data_dir = os.path.abspath(data_dir)
        assert os.path.isdir(data_dir), f"Invalid path {data_dir}"
        self.transform = kwargs.get("transform", None)

        # Поиск COCO аннотаций в директории
        self.dicom_paths = []

        coco_path = None
        for root, _, files in os.walk(data_dir):

            for file in files:
                basename = os.path.basename(file)
                filename, ext = os.path.splitext(basename)
                if ext == ".json":
                    coco_path = os.path.join(data_dir, root, file)
                elif ext in ["", ".dcm"]:
                    dicom_path = os.path.join(data_dir, root, file)
                    if pydicom.misc.is_dicom(dicom_path):
                        self.dicom_paths += [dicom_path]
        if not coco_path:
            raise FileNotFoundError(
                f"COCO аннотации не найдены в директории {data_dir}."
            )

        if not self.dicom_paths:
            raise FileNotFoundError(f"Dicom не найдены в директории {data_dir}.")

        # Загрузка COCO аннотаций
        with open(coco_path, "r") as f:
            self.coco = json.load(f)
        self.categories = self.coco["categories"]
        self.annotations = self.coco["annotations"]
        self.num_classes = len(self.categories)

        self.images = self.coco["images"]
        self.image_id_to_annotations = {image["id"]: [] for image in self.images}
        for ann in self.annotations:
            self.image_id_to_annotations[ann["image_id"]].append(ann)

        if len(self.images) != len(self.dicom_paths):
            new_images = []
            for img in self.images:
                for dicom_path in self.dicom_paths:
                    if dicom_path.endswith(img["file_name"]):
                        new_images += [img]
            self.images = new_images

        import re

        def extract_digits(s):
            out = re.findall(r"\d+", s)
            out = "".join(out)
            return int(out)

        self.images.sort(key=lambda x: extract_digits(x["file_name"]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:
            A dictionary with keys
             "image": image
             "mask": mask
             "path": dicom_path
             "raw_image": dicom_image


        """
        image_info = self.images[idx]
        for dicom_path in self.dicom_paths:
            if dicom_path.endswith(image_info["file_name"]):
                break
        else:
            raise FileNotFoundError(f"Dicom {dicom_path} не найден.")
        dicom = pydicom.dcmread(dicom_path)
        image = dicom_to_raster(dicom)

        anns = self.image_id_to_annotations[image_info["id"]]
        mask = self.get_mask(anns, image_info)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        raw = dicom.pixel_array

        if type(image) == torch.Tensor:
            image = image.float()
            shape = image.shape[1:]
            add_raw = False
        else:
            shape = image.shape[:2]
            add_raw = True

        out = {"image": image, "mask": mask, "path": dicom_path}

        if add_raw:
            if raw.shape[:2] != shape:
                # no need to apply all transforms
                raw = cv2.resize(raw, shape)
            out["raw_image"] = raw
        return out

    def get_mask(self, anns, image_info):
        mask = np.zeros(
            (image_info["height"], image_info["width"], self.num_classes),
            dtype=np.uint8,
        )
        for ann in anns:
            segmentation = ann["segmentation"]
            category_id = (
                ann["category_id"] - 1
            )  # Приведение category_id к индексу слоя
            if isinstance(segmentation, list):  # полигональная аннотация
                for polygon in segmentation:
                    poly_mask = self._polygon_to_mask(
                        polygon, image_info["height"], image_info["width"]
                    )
                    mask[:, :, category_id][poly_mask > 0] = 1
        return mask

    @staticmethod
    def _polygon_to_mask(polygon, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.array(polygon).reshape(-1, 2)
        mask = cv2.fillPoly(mask, [polygon.astype(int)], color=1)
        return mask

    def setup_infer(self):
        pass

    def infer_dataloader(self):
        return self

    def predict_dataloader(self):
        return self
