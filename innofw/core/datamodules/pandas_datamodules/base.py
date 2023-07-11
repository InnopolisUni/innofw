# standard libraries
from abc import ABC
from pathlib import Path
import pandas as pd

from innofw.core.datamodules.base import BaseDataModule
from innofw.utils.dm_utils.utils import find_file_by_ext

# local modules


class BasePandasDataModule(BaseDataModule, ABC):
    """
    A Base Class used for working with datasets in table formats
    ...

    Attributes
    ----------
    train_dataset : Path
        A path to train file
    test_dataset : Path
        A path to test file
    predict_dataset : Path
        A path to file for inference
    target_col : str
        The name of a column for prediction

    Methods
    -------
    setup_infer():
        The method prepares inference data

    """

    def __init__(
        self,
        train,
        test,
        target_col,
        infer=None,
        stage=None,
        *args,
        **kwargs,
    ):
        super().__init__(train, test, infer, stage=stage, *args, **kwargs)

        ext = ".csv"
        if not hasattr(self, "train_dataset"):
            self.train_dataset = (
                self.train if self.train is None else find_file_by_ext(self.train, ext)
            )
        if not hasattr(self, "test_dataset"):
            self.test_dataset = (
                self.test if self.test is None else find_file_by_ext(self.test, ext)
            )
        if not hasattr(self, "predict_dataset"):
            self.predict_dataset = (
                self.infer if self.infer is None else find_file_by_ext(self.infer, ext)
            )
        self.target_col = target_col

    def setup_infer(self):
        try:
            if isinstance(self.predict_dataset, str) or isinstance(
                self.predict_dataset, Path
            ):
                self.predict_dataset = pd.read_csv(self.predict_dataset)
        except Exception as err:
            raise FileNotFoundError(f"Could not read csv file: {err}")
