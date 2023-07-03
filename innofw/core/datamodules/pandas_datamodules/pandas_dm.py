# standard libraries
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.pandas_datamodules.base import (
    BasePandasDataModule,
)

# third party libraries
# local modules


class PandasDataModule(BasePandasDataModule):
    """
    A Class used for working with datasets in table formats for classification task
    ...

    Attributes
    ----------

    Methods
    -------
    setup_infer():
        The method prepares inference data

    save_preds(self, preds, stage: Stages, dst_path: Path):
        Saves inference predictions in CSV format
    """

    task = ["table-classification", "table-regression"]
    framework = [Frameworks.sklearn, Frameworks.xgboost, Frameworks.catboost]

    def __init__(
        self,
        train=None,
        test=None,
        target_col: Optional[str] = None,
        infer=None,
        stage=None,
        val_size: float = 0.2,
        augmentations=None,
        batch_size=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            train,
            test,
            target_col,
            infer=infer,
            stage=stage,
            *args,
            **kwargs,
        )
        self.setup(stage)

    def setup_train_test_val(self):
        try:
            if isinstance(self.train_dataset, str) or isinstance(
                self.train_dataset, Path
            ):
                self.train_dataset = pd.read_csv(self.train_dataset)
                self.test_dataset = pd.read_csv(self.test_dataset)
        except Exception as err:
            raise FileNotFoundError(f"Could not read csv file: {err}")

    def _get_x_n_y(self, dataset, target_col):
        result = {}
        if target_col is None:
            result["x"] = dataset
            result["y"] = None
        else:
            result["x"] = dataset.loc[:, dataset.columns != self.target_col]
            result["y"] = dataset[target_col]
        return result

    def train_dataloader(self):
        return self._get_x_n_y(self.train_dataset, self.target_col)

    def test_dataloader(self):
        return self._get_x_n_y(self.test_dataset, self.target_col)

    def predict_dataloader(self):
        return self._get_x_n_y(self.predict_dataset, self.target_col)

    def save_preds(self, preds, stage: Stages, dst_path: Path):
        df = self.get_stage_dataloader(stage)["x"]
        if self.target_col is None:
            df["y"] = preds
        else:
            df[self.target_col] = preds

        dst_filepath = Path(dst_path) / "prediction.csv"
        df.to_csv(dst_filepath)
        logging.info(f"Saved results to: {dst_filepath}")


class RegressionPandasDataModule(BasePandasDataModule):
    """
    A Class used for working with datasets in table formats for regression task
    ...

    Attributes
    ----------

    Methods
    -------
    setup_infer():
        The method prepares inference data

    save_preds(self, preds, stage: Stages, dst_path: Path):
        Saves inference predictions in CSV format
    """

    task = ["table-regression"]
    framework = [Frameworks.sklearn, Frameworks.xgboost, Frameworks.catboost]

    def __init__(
        self,
        train=None,
        test=None,
        target_col: Optional[str] = None,
        infer=None,
        flag=None,
        augmentations=None,
        batch_size=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            train,
            test,
            target_col,
            infer=infer,
            *args,
            **kwargs,
        )
        self.setup(flag)

    def setup_train_test_val(self):
        try:
            if isinstance(self.train_dataset, str) or isinstance(
                self.train_dataset, Path
            ):
                self.train_dataset = pd.read_csv(self.train_dataset)
                self.test_dataset = pd.read_csv(self.test_dataset)
        except Exception as err:
            raise FileNotFoundError(f"Could not read csv file: {err}")

    def _get_x_n_y(self, dataset, target_col):
        result = {}
        if target_col is None:
            result["x"] = dataset
            result["y"] = None
        else:
            result["x"] = dataset.loc[:, dataset.columns != self.target_col]
            result["y"] = dataset[target_col]
        return result

    def train_dataloader(self):
        return self._get_x_n_y(self.train_dataset, self.target_col)

    def test_dataloader(self):
        return self._get_x_n_y(self.test_dataset, self.target_col)

    def predict_dataloader(self):
        return self._get_x_n_y(self.predict_dataset, self.target_col)

    def save_preds(self, preds, stage: Stages, dst_path: Path):
        df = self.get_stage_dataloader(stage)["x"]
        if self.target_col is None:
            df["y"] = preds
        else:
            df[self.target_col] = preds

        dst_filepath = Path(dst_path) / "regression.csv"
        df.to_csv(dst_filepath)
        logging.info(f"Saved results to: {dst_filepath}")


class ClusteringPandasDataModule(BasePandasDataModule):
    """
    A Class used for working with datasets in table formats for clustering task
    ...

    Attributes
    ----------

    Methods
    -------
    setup_infer():
        The method prepares inference data

    save_preds(self, preds, stage: Stages, dst_path: Path):
        Saves inference predictions in CSV format
    """

    task = ["table-clustering"]
    framework = [Frameworks.sklearn, Frameworks.xgboost, Frameworks.catboost]

    def __init__(
        self,
        train=None,
        test=None,
        target_col: Optional[str] = None,
        infer=None,
        flag=None,
        augmentations=None,
        batch_size=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            train,
            test,
            target_col,
            infer=infer,
            *args,
            **kwargs,
        )
        self.setup(flag)

    def setup_train_test_val(self):
        # try:
        if isinstance(self.train_dataset, str) or isinstance(self.train_dataset, Path):
            self.train_dataset = pd.read_csv(self.train_dataset)
            self.test_dataset = pd.read_csv(self.test_dataset)
        # except Exception as err:
        #     raise FileNotFoundError(
        #         f"Could not read csv file: {self.test_dataset} error: {err}"
        #     )

    def _get_x_n_y(self, dataset, target_col):
        result = {}
        if target_col is None:
            result["x"] = dataset
            result["y"] = None
        else:
            result["x"] = dataset.loc[:, dataset.columns != self.target_col]
            result["y"] = dataset[target_col]
        return result

    def train_dataloader(self):
        return self._get_x_n_y(self.train_dataset, self.target_col)

    def test_dataloader(self):
        return self._get_x_n_y(self.test_dataset, self.target_col)

    def predict_dataloader(self):
        return self._get_x_n_y(self.predict_dataset, self.target_col)

    def save_preds(self, preds, stage: Stages, dst_path: Path):
        df = self.get_stage_dataloader(stage)["x"]
        if self.target_col is None:
            df["y"] = preds
        else:
            df[self.target_col] = preds

        dst_filepath = Path(dst_path) / "clustering.csv"
        df.to_csv(dst_filepath)
        logging.info(f"Saved results to: {dst_filepath}")
