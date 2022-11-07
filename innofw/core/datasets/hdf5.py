#
from pathlib import Path
from typing import List, Union

#
import albumentations as albu
import h5py
from torch.utils.data import Dataset

#
from .utils import prep_data


def get_indices_dict_size(hdf5_files):
    indices_dict_ = {}
    last_idx = 0
    for file in hdf5_files:
        with h5py.File(file) as f:
            last_idx += f["len"][0]
            indices_dict_[file] = last_idx

    return indices_dict_, last_idx


class HDF5Dataset(Dataset):
    """
        A class to represent a custom HDF5 Dataset.

        hdf5_files: Union[List[Path], List[str]]
            directory containing images
        bands_num : int
            number of bands in one image
        transform: albu.Compose


        Methods
        -------
        __getitem__(self, idx):
            returns transformed image and mask
    """
    def __init__(
        self,
        hdf5_files: Union[List[Path], List[str]],
        bands_num: int,
        transform: albu.Compose,
    ):
        self.indices_dict, self.size_ = get_indices_dict_size(hdf5_files)
        self.bands_num = bands_num
        self.transform = transform

    def __len__(self):
        return self.size_

    def __getitem__(self, idx):
        count = 0
        for key, value in self.indices_dict.items():
            if idx <= value:
                with h5py.File(key) as f:
                    index = idx - count

                    image = f[str(index)][..., : self.bands_num]
                    mask = f[str(index)][..., self.bands_num :]

                    return prep_data(image, mask, self.transform)
            else:
                count += value
        raise Exception(f"Dataset element {idx} not found; Max is {self.__len__()}")
