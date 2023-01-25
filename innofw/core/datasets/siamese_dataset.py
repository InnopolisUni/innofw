import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from innofw.core.augmentations import Augmentation


class SiameseDataset(Dataset):
    """
     A class to represent a Siamese Dataset.

     ...

     Attributes
     ----------
     data_path : str
         path to directory containing folders with images
     transform : Iterable[albumentations.augmentations.transforms]


     Methods
     -------
     __getitem__(self, index: int):
         returns image1, image2 and information about their belonging to the same class

    read_img(self, path):
         reads an image using PIL
    """

    def __init__(self, data_path, transform):
        classes = []
        images = []
        for folder in os.listdir(data_path):
            path = os.path.join(data_path, folder)
            for image in os.listdir(path):
                images.append(os.path.join(path, image))
                classes.append(folder)
        encoder = {}
        for i, label in enumerate(os.listdir(data_path)):
            encoder[label] = i

        self.datapath = data_path
        self.transform = Augmentation(transform)
        self.encoder = encoder
        self.classes = classes
        self.images = images

    def __getitem__(self, idx):
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            current_class = self.classes[random.randint(0, len(self.classes) - 1)]
            class_path = os.path.join(self.datapath, current_class)
            imgs = os.listdir(class_path)

            image1 = self.read_img(
                os.path.join(class_path, imgs[random.randint(0, len(imgs) - 1)])
            )
            image2 = self.read_img(
                os.path.join(class_path, imgs[random.randint(0, len(imgs) - 1)])
            )
        else:
            current_classes = random.sample(self.classes, k=2)
            class_path = os.path.join(self.datapath, current_classes[0])
            imgs = os.listdir(class_path)
            image1 = self.read_img(
                os.path.join(class_path, imgs[random.randint(0, len(imgs) - 1)])
            )

            class_path = os.path.join(self.datapath, current_classes[1])
            imgs = os.listdir(class_path)
            image2 = self.read_img(
                os.path.join(class_path, imgs[random.randint(0, len(imgs) - 1)])
            )

        return (
            image1,
            image2,
            torch.from_numpy(
                np.array([int(should_get_same_class == 0)], dtype=np.float32).copy()
            ),
        )

    def __len__(self):
        return len(self.images)

    def read_img(self, img_path):
        image = Image.open(img_path)
        image = np.asarray(image)
        image = torch.from_numpy(image.copy())
        if self.transform:
            image = self.transform(image)
        return image


class SiameseDatasetInfer(Dataset):
    """
    A class to represent a Siamese Dataset.

    ...

    Attributes
    ----------
    data_path : str
        path to directory containing folders with images
    transform : Iterable[albumentations.augmentations.transforms]


    Methods
    -------
    __getitem__(self, index: int):
        returns image1, image2

    read_img(self, path):
        reads an image using PIL
    """

    def __init__(self, data_path, transform):
        self.datapath = data_path
        self.transform = transform

        images = []
        image_pairs = []
        image_pair_names = []
        for image in os.listdir(data_path):
            images.append(os.path.join(data_path, image))

        for ind, image in enumerate(images):
            for image2 in images[ind:]:
                image_pairs.append((self.read_img(image), self.read_img(image2)))
                image_pair_names.append((image, image2))

        self.images = images
        self.image_pairs = image_pairs
        self.image_pair_names = image_pair_names

    def __getitem__(self, idx):
        return self.image_pairs[idx]

    def __len__(self):
        return len(self.image_pairs)

    def read_img(self, img_path):
        image = Image.open(img_path)
        image = np.asarray(image)
        image = torch.from_numpy(image.copy())
        if self.transform:
            image = self.transform(image)
        return image
