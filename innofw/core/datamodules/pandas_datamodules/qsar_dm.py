import logging
from pathlib import Path

import pandas as pd

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.pandas_datamodules.base import (
    BasePandasDataModule,
)
from innofw.core.datasets.smiles_dataset import SmilesDataset


class QsarDataModule(BasePandasDataModule):
    """
    DataModule for smiles data.

    Attributes
    ----------
    smiles_col : str
        Specify the column name that contains the target values
    target_col : str
        Specify the column name that contains the target values
    task : list
        List if supported tasks
    framework : list
        List of supported frameworks

    Methods
    -------
    setup_train_test_val():
        The setup_train_test_val function is called by the DataModule class to set up the training and test datasets.
        It reads in a csv file containing smiles strings and target values, then splits it into train/test sets.
        The SmilesDataset class is used to create a PyTorch dataset object for each of these sets.
    """

    task = ["qsar-classification", "qsar-regression"]
    framework = [Frameworks.sklearn, Frameworks.xgboost, Frameworks.catboost]
    train_smiles_dataset: SmilesDataset = None
    test_smiles_dataset: SmilesDataset = None
    lower_bound: float = None
    upper_bound: float = None
    _preprocessed: bool = False

    def __init__(
        self,
        train,
        test,
        smiles_col: str,
        target_col,
        infer=None,
        target_value=None,
        delta_precision=0,
        stage=None,
        val_size: float = 0.2,
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
        self.smiles_col = smiles_col
        if target_value is not None:
            self.lower_bound = target_value - delta_precision
            self.upper_bound = target_value + delta_precision

        if self.train is None:
            self.train_dataset = self._get_data(train)
            if isinstance(self.train_dataset, tuple):
                self.train_dataset = self.train_dataset[0]
        if self.test is None:
            self.test_dataset = self._get_data(test)
            if isinstance(self.test_dataset, tuple):
                self.test_dataset = self.test_dataset[0]
        self.setup(stage)

    def setup_train_test_val(self):
        if self._preprocessed:
            return

        if isinstance(self.train_dataset, str) or isinstance(self.train_dataset, Path):
            train_csv = pd.read_csv(self.train_dataset)
            test_csv = pd.read_csv(self.test_dataset)

        train_smiles, train_target = (
            train_csv[self.smiles_col].values,
            train_csv[self.target_col].values,
        )
        test_smiles, test_target = (
            test_csv[self.smiles_col].values,
            test_csv[self.target_col].values,
        )

        self.train_smiles_dataset = SmilesDataset(
            train_smiles, train_target, self.target_col
        )
        self.test_smiles_dataset = SmilesDataset(
            test_smiles, test_target, self.target_col
        )
        self._preprocessed = True

    def setup_infer(self):
        self.setup_train_test_val()
        predict_csv = pd.read_csv(self.predict_dataset)
        predict_smiles = predict_csv[self.smiles_col]
        self.predict_smiles_dataset = SmilesDataset(
            predict_smiles, [None] * len(predict_smiles), self.target_col
        )

    def train_dataloader(self):
        x = pd.DataFrame(self.train_smiles_dataset.X)
        y = self.train_smiles_dataset.y
        return {"x": x, "y": y}

    def test_dataloader(self):
        x = pd.DataFrame(self.test_smiles_dataset.X)
        y = self.test_smiles_dataset.y
        return {"x": x, "y": y}

    def predict_dataloader(self):
        x = pd.DataFrame(self.predict_smiles_dataset.X)
        return {"x": x}

    def get_stage_dataloader(self, stage):
        if stage is Stages.train:
            return self.train_dataloader()
        elif stage is Stages.test:
            return self.test_dataloader()
        elif stage is Stages.predict:
            return self.predict_dataloader()
        else:
            raise ValueError("Wrong stage passed use on of following:", list(Stages))

    def save_preds(self, preds, stage: Stages, dst_path: Path):
        df = pd.read_csv(
            getattr(self, f"{stage.name}_dataset"), usecols=[self.smiles_col]
        )
        if preds.ndim == 2:
            preds = preds[:, 0]
        if self.target_col is None:
            df["y"] = preds
        else:
            df[self.target_col] = preds
        if self.lower_bound is not None and self.upper_bound is not None:
            df = df[
                (self.lower_bound <= df[self.target_col])
                & (df[self.target_col] <= self.upper_bound)
            ]
        dst_filepath = Path(dst_path) / "preds.csv"
        df.to_csv(dst_filepath, index=False)
        logging.info(f"Saved results to: {dst_filepath}")
