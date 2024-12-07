import json
import os

import cv2
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

from innofw.utils.data_utils.preprocessing.dicom_handler import dicom_to_raster


class DicomCocoDatasetRTK(Dataset):
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
            # raise FileNotFoundError(
            print(f"COCO аннотации не найдены в директории {data_dir}.")
            self.coco_found = False
        else:
            self.coco_found = True

        if not self.dicom_paths:
            raise FileNotFoundError(f"Dicom не найдены в директории {data_dir}.")

        import re

        def extract_digits(s):
            out = re.findall(r"\d+", s)
            out = "".join(out)
            return int(out)

        # Загрузка COCO аннотаций
        if self.coco_found:
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

            self.images.sort(key=lambda x: extract_digits(x["file_name"]))
        else:
            self.dicom_paths.sort()

    def __len__(self):
        if self.coco_found:
            return len(self.images)
        else:
            return len(self.dicom_paths)

    def get_dicom(self, i):

        dicom_path = self.dicom_paths[i]
        dicom = pydicom.dcmread(dicom_path)
        image = dicom_to_raster(dicom)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        if type(image) == torch.Tensor:
            image = image.float()

        out = {"image": image, "path": dicom_path}
        return out

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
        if not self.coco_found:
            return self.get_dicom(idx)
        image_info = self.images[idx]
        for dicom_path in self.dicom_paths:
            if dicom_path.endswith(image_info["file_name"]):
                break
        else:
            print(self.dicom_paths, image_info["file_name"])
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
