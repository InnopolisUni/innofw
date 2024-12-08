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

    def _get_x_n_y(self, dataset, target_col):
        return {
            "x": dataset[TEXT_DATA_COLUMN_NAME],
            "y": None if target_col is None else dataset[target_col].str.strip(),
        }
