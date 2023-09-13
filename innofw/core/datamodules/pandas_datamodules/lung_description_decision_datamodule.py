import logging

from pathlib import Path
import pandas as pd

from innofw.constants import Stages
from innofw.core.datamodules.pandas_datamodules.pandas_dm import PandasDataModule


class LungDescriptionDecisionPandasDataModule(PandasDataModule):
    def save_preds(self, preds, stage: Stages, dst_path: Path):
        df = self.get_stage_dataloader(stage)["x"]
        df = pd.DataFrame(df)
        if self.target_col is None:
            df["y"] = preds
        else:
            df[self.target_col] = preds

        dst_filepath = Path(dst_path) / "prediction.csv"
        df.to_csv(dst_filepath)
        logging.info(f"Saved results to: {dst_filepath}")

    def _get_x_n_y(self, dataset, target_col):
        result = {}
        result["x"] = dataset["SR"]
        if target_col is None:
            result["y"] = None
        else:
            result["y"] = dataset[target_col]
        return result
