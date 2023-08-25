from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from torchvision.io import read_image


def cached_dataset_feature(cls):
    class WrappedDataset(Dataset):
        def __init__(self, *args, **kwargs):
            self._use_caching = False
            self.ds = cls(*args, **kwargs)

            self.container = [self.ds[i] for i in range(len(self.ds))]

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, index: Any):
            if (
                self._use_caching
                and self.container is not None
                and len(self.container) != 0
            ):
                return self.container[index]
            return self.ds[index]

        @property
        def use_caching(self):
            return self._use_caching

        @use_caching.setter
        def use_caching(self, use_caching):
            if use_caching:
                self._use_caching = True
                self.container = (
                    [] if self.container is None else self.container
                )
            else:
                self._use_caching = False
                del self.container
                self.container = []

    return WrappedDataset


@cached_dataset_feature
class CustomImageDataset(Dataset):
    def __init__(self, img_dir):
        self.files = list(Path(img_dir).rglob("*.PNG"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = read_image(str(self.files[idx]))
        return image


# so any other dataset can be extended like this
