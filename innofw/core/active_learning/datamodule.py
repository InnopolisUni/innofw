from typing import Protocol
from typing import TypedDict

import numpy as np
import pandas as pd
from abc import abstractmethod 



class DataContainer(TypedDict):
    """
    A DataContainer key set for typing.

    Attributes
    ----------
    x : pd.DataFrame
        DataFrame with data
    y : np.ndarray
        numpy array with targets
    """

    x: pd.DataFrame
    y: np.ndarray


class DataModuleI(Protocol):
    """
    A DataModuleI interface for using in ActiveDataModule.

    Methods
    -------
    test_dataloader():
        returns DataContainer instance.
    train_dataloader():
        returns DataContainer instance.
    setup():
        returns Nothing.
    """
    @abstractmethod
    def test_dataloader(self) -> DataContainer: # pragma: no cover
        ...
    @abstractmethod
    def train_dataloader(self) -> DataContainer: # pragma: no cover
        ...
    @abstractmethod
    def setup(self) -> None: # pragma: no cover
        ...


class ActiveDataModule:
    """
    A DataModule wrapper class for active learning.

    Attributes
    ----------
    datamodule : DataModuleI
        first name of the person
    init_size : float
        size of initial training set

    Methods
    -------
    update_indices(indices):
        update train indexes from pool.
    """

    _preprocessed: bool = False

    def __init__(self, datamodule: DataModuleI, init_size: float = 0.1):
        self.datamodule = datamodule
        self.init_size = init_size
        self.setup()

    def update_indices(self, indices):
        idx_to_add = self.pool_idxs[indices]
        self.pool_idxs = np.setdiff1d(self.pool_idxs, idx_to_add)
        self.train_idxs = np.concatenate(
            [self.train_idxs, idx_to_add], axis=None
        )

    def train_dataloader(self):
        return {
            "x": self.train_dl["x"].iloc[self.train_idxs],
            "y": self.train_dl["y"][self.train_idxs],
        }

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def pool_dataloader(self):
        return {
            "x": self.train_dl["x"].iloc[self.pool_idxs],
        }

    def setup(self):
        if self._preprocessed:
            return

        self.datamodule.setup()

        self.train_dl = self.datamodule.train_dataloader()
        self.train_idxs = np.random.choice(
            len(self.train_dl["y"]),
            size=int(len(self.train_dl["y"]) * self.init_size),
            replace=False,
        )
        self.pool_idxs = np.setdiff1d(
            np.arange(len(self.train_dl["y"])), self.train_idxs
        )
        self._preprocessed = True


def get_active_datamodule(datamodule):
    return ActiveDataModule(datamodule)
