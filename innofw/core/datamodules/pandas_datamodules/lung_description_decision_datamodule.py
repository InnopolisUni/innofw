import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from innofw.constants import Stages
from innofw.core.datamodules.pandas_datamodules.pandas_dm import PandasDataModule

TEXT_DATA_COLUMN_NAME = "SR"


class LungDescriptionDecisionPandasDataModule(PandasDataModule):
    """Datamodule for processing data

    As input any csv data may be considered. This datamodule retrieves "SR" column and
    sends it to module as pd.Series, cause pd.DataFrame is not supported for
    the given sklearn pipeline. "SR" stands for "Structured report" the field in DICOM
    """

    def save_preds(self, preds: np.ndarray, stage: Stages, dst_path: Path):
        """saving result as csv file

        The main difference is that sklearn pipeline gets as input pd.Series data,
        while standard datamodule retrieves pd.DataFrame, thus this method merge
        input which is pd.Series and output (np.ndarray) and saves as csv file.

        :param preds: result of sklearn pipeline
        :param stage:
        :param dst_path:
        """
        df = pd.DataFrame(self.get_stage_dataloader(stage)["x"])
        if self.target_col is None:
            df["y"] = preds
        else:
            df[self.target_col] = preds

        if self.infer:
            dst_path = os.path.dirname(self.infer)
        dst_filepath = Path(dst_path) / "prediction.csv"
        df.to_csv(dst_filepath)
        logging.info(f"Saved results to: {dst_filepath}")

    def _get_x_n_y(self, dataset, target_col):
        return {"x": dataset[TEXT_DATA_COLUMN_NAME], "y": None if target_col is None else dataset[target_col].str.strip()}
