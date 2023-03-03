import math
import sys
import os
import glob
import random
from typing import List

from functools import reduce, partial
import albumentations as albu
import numpy as np

import torch
import torch.nn as nn
import torchvision

import h5py
from aeronet.dataset import BandCollection, parse_directory
from tqdm import tqdm as tqdm


def _to_tensor(x, **kwargs):
    if x.shape[0] == 3 or x.shape[0] == 4:
        return x.astype('float32')

    return x.transpose(2, 0, 1).astype('float32')

def _get_preprocessing_fn(mean=None, std=None):
    def preprocess_input(x, mean, std, **kwargs):
        if mean is not None:
            mean = np.array(mean)
            # x = x - mean
        if std is not None:
            std = np.array(std)
            # x = x / std
        return x

    return partial(preprocess_input, mean=mean, std=std)

def _augment_and_preproc(image, mask, augmentations, preprocessing):
    if augmentations is not None:
        image, mask = augmentations(image, mask)  # image=image, mask=mask
        #  = augmented['image']
        # mask = augmented['mask']

    if preprocessing is not None:
        image = preprocessing(image)

#         assert self.path_to_hdf5.endswith('NIR_B11_Red_Green_1-11.hdf5')
#         orig = image.copy()
#         eps = 1e-6
#         image[:,:,0] = (orig[:,:,0] - orig[:,:,2] + eps) / (orig[:,:,0] + orig[:,:,2] + eps) # NDVI
#         image[:,:,1] = (orig[:,:,3] - orig[:,:,0] + eps) / (orig[:,:,3] + orig[:,:,0] + eps) # NDWI
#         image[:,:,2] = (orig[:,:,3] - orig[:,:,1] + eps) / (orig[:,:,3] + orig[:,:,1] + eps) # MNDWI
#         image[:,:,3] = (orig[:,:,1] - orig[:,:,0] + eps) / (orig[:,:,1] + orig[:,:,0] + eps) # NDBI

    # convert to tensor shape
    image = _to_tensor(image)
    mask = _to_tensor(mask)

    mask[mask == 255] = 1

    return image, mask

def _get_class_weights(arr):
    labels = np.array(arr)
    uniq_values = np.unique(labels)
    class_weights = np.zeros(labels.shape)
    for v in uniq_values:
        x = (labels == v).astype('float32')
        p = np.sum(x) / len(labels)
        class_weights = np.where(
            x > 0,
            # '0' - background, '1' - quarries, '2' - fp quarries (also background)
            (1 / p if v == 1 else 0.75 / p if v == 0 else 0.25 / p) if len(uniq_values) == 3 else \
            1 - p if len(uniq_values) == 2 else \
            1 / p,
#             1 / p if v == 1 else 1 / (p * (len(uniq_values) - 1)),
            class_weights
        )
    return class_weights

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_hdf5,
                 in_channels=3,
                 augmentations=None,
                 preprocessing=None,
                 with_mosaic=False):
        self.path_to_hdf5 = path_to_hdf5
        from innofw.core.augmentations import Augmentation
        self.augmentations = Augmentation(augmentations)
        self.preprocessing = preprocessing
        self.in_channels = in_channels
        self.mean = None
        self.std = None
        self.with_mosaic = with_mosaic

        with h5py.File(self.path_to_hdf5, 'r') as f:
            self.len = f['len'][0]
            try:
                self.class_ids = f['class_ids'][:]
                self.class_weights = _get_class_weights(self.class_ids)
            except Exception as e:
                self.class_weights = f['class_weights'][:]
                self.class_ids = (self.class_weights > 1).astype('float32')
            assert len(self.class_ids) == len(self.class_weights)

            print('dataset:', self.path_to_hdf5)
            print('class_ids unique:', np.unique(self.class_ids))
            print('class_weights unique:', np.unique(self.class_weights))

            try:
                self.mean = f['mean'][:]
                self.std = f['std'][:]
            except Exception as e:
                # default RGB-values for hdf5 without corresponding data
                self.mean = [0.03935977, 0.06333545, 0.07543217]
                self.std = [0.00962875, 0.01025132, 0.00893212]
#             assert len(self.mean) == self.in_channels
#             assert len(self.mean) == len(self.std)

        self.norm_class_weights = self.class_weights / self.class_weights.sum()

        if self.preprocessing is None:
            self.preprocessing = _get_preprocessing_fn(self.mean, self.std)
        else:
            self.mean = self.preprocessing.keywords['mean']
            self.std = self.preprocessing.keywords['std']
#         assert (self.mean is None or \
#                 len(self.mean) == self.in_channels)
#         assert self.len == len(self.class_weights)

    def _get_item_data(self, index):
        with h5py.File(self.path_to_hdf5, 'r') as f:
            data = f[str(index)]
            image = data[:,:,:self.in_channels]
            mask = data[:,:,self.in_channels:]
            try:
                raster_name = f[str(index) + '_raster_name'][()].decode("utf-8")
                geo_bounds = f[str(index) + '_geo_bounds'][:]
                assert len(geo_bounds) == 4
                meta = {'raster_name': raster_name, 'geo_bounds': geo_bounds}
            except Exception:
                meta = {}

        assert len(image.shape) == len(mask.shape)
        assert image.shape[:2] == mask.shape[:2]

        return image, mask, meta

    def __getitem__(self, index):
        image, mask, meta = self._get_item_data(index)

        # https://www.kaggle.com/nvnnghia/awesome-augmentation
        if self.with_mosaic and random.randint(0, 1):
            s = image.shape[:2]
            h, w = image.shape[:2]
            # indices = [index] + [np.random.randint(0, self.len) for _ in range(3)]
            indices = [index] + [ind for ind in np.random.choice(self.len, size=3, p=self.norm_class_weights)]
            for i, ind in enumerate(indices):
                if i > 0:
                    image, mask, _ = self._get_item_data(ind)
                img, msk = image, mask
                if i == 0: # top left
                    xc, yc = [int(random.uniform(s[0] * 0.5, s[1] * 1.5)) for _ in range(2)]  # mosaic center x, y
                    img4 = np.full((s[0] * 2, s[1] * 2, img.shape[2]), 0, dtype=np.float32)  # base image with 4 tiles
                    msk4 = np.full((s[0] * 2, s[1] * 2, msk.shape[2]), 0, dtype=np.float32)  # base mask with 4 tiles
                    x1a, y1a, x2a, y2a = 0, 0, xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - xc, h - yc, w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, 0, s[1] * 2, yc
                    x1b, y1b, x2b, y2b = 0, h - yc, x2a - x1a, h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = 0, yc, xc, s[0] * 2
                    x1b, y1b, x2b, y2b = w - xc, 0, w, y2a - y1a
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, s[1] * 2, s[0] * 2
                    x1b, y1b, x2b, y2b = 0, 0, x2a - x1a, y2a - y1a

                yb = np.abs(np.arange(y1b, y2b))
                xb = np.abs(np.arange(x1b, x2b))

                # 101-reflection for indices greater or equal than h (or w)
                # transform [..., h-3, h-2, h-1, h+0, h+1, ...] to
                #           [..., h-3, h-2, h-1, h-2, h-3, ...]
                bad_ybi = np.where(yb >= h)[0]
                if bad_ybi.any():
                    fixed_ybi = [y - 2 * (i + 1) for i, y in enumerate(bad_ybi)]
                    yb[bad_ybi] = yb[fixed_ybi]

                bad_xbi = np.where(xb >= w)[0]
                if bad_xbi.any():
                    fixed_xbi = [x - 2 * (i + 1) for i, x in enumerate(bad_xbi)]
                    xb[bad_xbi] = xb[fixed_xbi]

                img4[y1a:y2a, x1a:x2a] = img[np.ix_(yb, xb)]
                msk4[y1a:y2a, x1a:x2a] = msk[np.ix_(yb, xb)]
            image, mask = img4, msk4

        image, mask = _augment_and_preproc(image, mask, self.augmentations, self.preprocessing)

        # d2 = (mask == 1).astype(np.int8)
        # d3 = (mask == 2).astype(np.int8)

        # mask = np.concatenate([d2,d3], axis=0)
#         mask = torch.cat([d2, d3], axis=1)


        from innofw.constants import SegDataKeys
        return {
            SegDataKeys.image: image,
            SegDataKeys.label: mask,
            # 'metadata': meta
        }

    def __len__(self):
        return self.len

class DatasetUnion(torch.utils.data.Dataset):
    def __init__(self, dataset_list):
        self.datasets = []
        self.class_ids = []
        self.len = 0

        for dataset in dataset_list:
            self.datasets.append(dataset)
            self.class_ids.extend(list(dataset.class_ids))
            self.len += len(dataset)
        self.class_weights = _get_class_weights(self.class_ids)

        print('DatasetUnion')
        print('class_ids unique:', np.unique(self.class_ids))
        print('class_weights unique:', np.unique(self.class_weights))

        mean_std_valid = [dataset.mean is not None and dataset.std is not None for dataset in dataset_list]
        assert all(mean_std_valid) or all(list(~np.array(mean_std_valid)))

        self.mean = None
        self.std = None
        if all(mean_std_valid):
            mean_s = [d.mean for d in dataset_list]
            std_s = [d.std for d in dataset_list]
            self.mean = np.mean(mean_s, axis=0)
            self.std = np.mean(std_s, axis=0)
        self.preprocessing = _get_preprocessing_fn(self.mean, self.std)

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            else:
                index -= len(dataset)
        raise Exception(f'DatasetUnion: dataset element {index} not found')

    def __len__(self):
        return self.len

class WeightedRandomCropDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tif_folders: List[str],
        crop_size=(224, 224),
        band_channels=['B04', 'B03', 'B02'],
        band_labels=['100'],
        augmentations=None,
        is_train=True,
        verbose=False
    ):
        self.tif_folders = tif_folders
        self.augmentations = augmentations
        self.preprocessing = None
        self.in_channels = len(band_channels)
        self.mean = None
        self.std = None
        self.verbose = verbose

        self.band_channels = band_channels
        self.band_labels = band_labels
        self.crop_size = crop_size
        self.samples_per_image = ((10980 + crop_size[0] - 1) // crop_size[0]) ** 2
        self.len = self.samples_per_image * len(self.tif_folders)

        self.is_train = is_train

        self.images = []
        self.masks = []
        self.sizes = []
        self.fg_indicies = []
        self.bg_indicies = []
        self.val_indices = []
        with tqdm(self.tif_folders, desc='WeightedRandomCrop dataset: ' + ('train' if is_train else 'val'), file=sys.stdout) as iterator:
            for tif_folder in iterator:
                image = BandCollection(parse_directory(tif_folder, self.band_channels)).ordered(*self.band_channels)
                mask = BandCollection(parse_directory(tif_folder, self.band_labels))

                self.images.append(image)
                self.masks.append(mask)
                self.sizes.append(mask.numpy().shape)

                m = mask.numpy().flatten()
                if self.is_train:
                    self.fg_indicies.append(np.where(m > 0)[0])
                    self.bg_indicies.append(np.where(m == 0)[0])
                else:
                    weights = _get_class_weights(mask.numpy().flatten())
                    weights /= weights.sum()
                    self.val_indices.append(np.random.choice(len(weights), self.samples_per_image, p=weights))

    def __getitem__(self, index):
        image_idx = index // self.samples_per_image

        image = self.images[image_idx]
        mask = self.masks[image_idx]
        _, height, width = self.sizes[image_idx]

        if self.is_train:
            fg_indicies = self.fg_indicies[image_idx]
            bg_indicies = self.bg_indicies[image_idx]
            if len(fg_indicies) > 0 and np.random.rand() < 0.5:
                fg_idx = np.random.randint(len(fg_indicies))
                crop_idx = fg_indicies[fg_idx]
            else:
                bg_idx = np.random.randint(len(bg_indicies))
                crop_idx = bg_indicies[bg_idx]
        else:
            crop_idx = self.val_indices[image_idx][index % self.samples_per_image]

        h_crop, w_crop = self.crop_size
        y_crop = crop_idx // width
        x_crop = crop_idx % width
        dy = np.random.randint(self.crop_size[0])
        dx = np.random.randint(self.crop_size[1])

        y_crop = np.max(y_crop - dy, 0)
        x_crop = np.max(x_crop - dx, 0)

        sample = image.sample(y_crop, x_crop, h_crop, w_crop)
        
        if self.verbose:
            coords = [x_crop, y_crop, w_crop, h_crop]
            print('coords:', coords)

        x_sample = sample.numpy().astype('float32')
        y_sample = mask.sample(y_crop, x_crop, h_crop, w_crop).numpy().astype('float32')

        image = x_sample.transpose(1, 2, 0)
        image[image < 0] = 0
        image[image > 1] = 1

        mask = y_sample.transpose(1, 2, 0)

        assert len(image.shape) == len(mask.shape)
        assert image.shape[:2] == mask.shape[:2]

        image, mask = _augment_and_preproc(image, mask, self.augmentations, self.preprocessing)

        return {
            'image': image,
            'mask': mask,
            'metadata': {
                'raster_name': self.tif_folders[image_idx],
                'geo_bounds': torch.tensor(sample.bounds[:])
            }
        }

    def __len__(self):
        return self.len


def _intersection(a, b):
    if a is None or b is None:
        return 0

    # [xl, yl, xr, yr]
    rect_a = np.array([a[0], a[1], a[0] + a[2] - 1, a[1] + a[3] - 1])
    rect_b = np.array([b[0], b[1], b[0] + b[2] - 1, b[1] + b[3] - 1])

    bl = np.max([rect_a, rect_b], axis=0)
    ur = np.min([rect_a, rect_b], axis=0)

    wh = ur[2:4] - bl[0:2] + 1

    return max(0, wh[0] * wh[1])

class _TiledQuarryImage(torch.utils.data.Dataset):
    def __init__(
        self,
        tif_folder: str,
        crop_size,
        crop_step,
        band_channels,
        band_labels,
        augmentations,
    ):
        if crop_step[0] < 1 or crop_step[0] > crop_size[0]:
            raise ValueError()
        if crop_step[1] < 1 or crop_step[1] > crop_size[1]:
            raise ValueError()

        self.tif_folder = tif_folder
        self.crop_size = crop_size
        self.crop_step = crop_step
        self.band_channels = band_channels
        self.band_labels = band_labels
        self.augmentations = augmentations
        self.mean = None
        self.std = None
        self.preprocessing = _get_preprocessing_fn(self.mean, self.std)

        self.image = BandCollection(parse_directory(tif_folder, self.band_channels))
        self.mask = BandCollection(parse_directory(tif_folder, self.band_labels))

        if self.image.shape[1:] != self.mask.shape[1:]:
            raise ValueError("Shape of image does not corresponds to shape of mask")

        _, rows, cols = self.mask.shape

        class_ids = []
        crops = []
        y_tiles = (rows - self.crop_size[0] + 1 + self.crop_step[0] - 1) // self.crop_step[0]
        x_tiles = (cols - self.crop_size[1] + 1 + self.crop_step[1] - 1) // self.crop_step[1]
        n_tiles = x_tiles * y_tiles

        skip = False
        with tqdm(range(n_tiles), desc='TiledQuarryImage', file=sys.stdout) as iterator:
            for n in iterator:
                y = (n // x_tiles) * self.crop_step[0]
                x = (n % x_tiles) * self.crop_step[1]
                crop = (x, y, self.crop_size[1], self.crop_size[0])
                prev_crop = crops[-1] if len(crops) > 0 else None
                s = _intersection(prev_crop, crop)
                if skip and s > 0:
                    continue
                crops.append(crop)
                mask = self.mask.sample(y, x, self.crop_size[0], self.crop_size[1]).numpy()
                m_sum = np.sum(mask)
                class_ids.append(1 if m_sum > 0 else 0)
                skip = not (m_sum > 0)


        self.crops = np.array(crops)
        self.class_ids = np.array(class_ids)
        self.class_weights = _get_class_weights(self.class_ids)

        print(np.unique(self.class_ids))
        print(np.unique(self.class_weights))

    def __getitem__(self, index):
        x, y, w, h = self.crops[index]

        mask = self.mask.sample(y, x, h, w).numpy().astype('float32')

        sample = self.image.sample(y, x, h, w)
        image = sample.numpy().astype('float32')

        image = image.transpose(1, 2, 0)
        mask = mask.transpose(1, 2, 0)

        assert len(image.shape) == len(mask.shape)
        assert image.shape[:2] == mask.shape[:2]

        image[image < 0] = 0
        image[image > 1] = 1
        image, mask = _augment_and_preproc(image, mask, self.augmentations, self.preprocessing)

        return {
            'image': image,
            'mask': mask,
            'metadata': {
                'raster_name': self.tif_folder,
                'geo_bounds': torch.tensor(sample.bounds[:])
            }
        }

    def __len__(self):
        return len(self.crops)

class TiledDataset(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        tif_folders: List[str],
        crop_size=(224, 224),
        crop_step=(224, 224),
        band_channels=['B04', 'B03', 'B02'],
        band_labels=['100'],
        augmentations=None,
    ):
        self.mean = None
        self.std = None
        self.preprocessing = _get_preprocessing_fn(self.mean, self.std)

        class_ids = []
        datasets = []
        for n, tif_folder in enumerate(tif_folders):
            print(f'Image [{n + 1}/{len(tif_folders)}]: {tif_folder}')
            dataset = _TiledQuarryImage(
                tif_folder, crop_size, crop_step, band_channels, band_labels, augmentations
            )
            class_ids.extend(list(dataset.class_ids))
            datasets.append(dataset)
        self.class_ids = np.array(class_ids)
        self.class_weights = _get_class_weights(self.class_ids)

        print('TiledDataset')
        print('class_ids unique:', np.unique(self.class_ids))
        print('p for 1-class:', np.sum(self.class_ids) / len(self.class_ids))
        print('class_weights unique:', np.unique(self.class_weights))

        super().__init__(datasets)
