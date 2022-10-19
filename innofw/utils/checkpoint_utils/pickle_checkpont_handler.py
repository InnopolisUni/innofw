from pathlib import Path
from typing import Union, Optional
import logging

import sklearn
from pickle import dump, load
from innofw.constants import CheckpointFieldKeys
from innofw.utils import get_abs_path

# from innoframework.schema.model_checkpoint import ModelCheckpoint  # todo: replace the checkpoint from dict to model_checkpoint type(+think about it more)
from innofw.utils.checkpoint_utils.base_checkpoint_handler import (
    CheckpointHandler,
)


class PickleCheckpointHandler(CheckpointHandler):
    @staticmethod
    def save_ckpt(
        model: sklearn.base.BaseEstimator,
        dst_path: Union[str, Path],
        metadata: Optional[dict] = None,
        file_extension: str = ".pickle",
        create_default_folder: bool = False,
        # todo: fix type of  the model as it can be not only BaseEstimator
    ) -> Path:
        """Saves a serialized model object into destination path with name best.pkl

        Arguments:
            model: model to save
            dst_path: path to store the model
            metadata: a dictionary with additional information to store along with model
            file_extension: extension of the file


        Resulting file contains two fields: model and metadata
        """
        dst_path = Path(dst_path)
        dst_path = get_abs_path(dst_path)

        if dst_path.suffix != file_extension:
            if create_default_folder and "checkpoints" not in dst_path.parts:
                dst_path /= "checkpoints"
            dst_path.mkdir(exist_ok=True, parents=True)
            dst_path /= f"model{file_extension}"

        try:
            inner_model = model[CheckpointFieldKeys.model]
            inner_metadata = model[CheckpointFieldKeys.metadata]

            if inner_metadata is None:
                inner_metadata = metadata.copy()
            else:
                inner_metadata.update(metadata)

            data = {
                CheckpointFieldKeys.model: inner_model,
                CheckpointFieldKeys.metadata: inner_metadata,
            }
        except:
            data = {
                CheckpointFieldKeys.model: model,
                CheckpointFieldKeys.metadata: metadata,
            }

        with open(dst_path, "wb+") as f:
            dump(data, f)

        logging.info(f"Saved a checkpoint at: {dst_path}")

        return dst_path

    @staticmethod
    def load_ckpt(ckpt_path: Union[str, Path]) -> dict:  # -> ModelCheckpoint:
        """Deserializes a model

        Arguments:
            ckpt_path: path to the serialized model
        """
        if not ckpt_path.is_absolute():
            ckpt_path = get_abs_path(ckpt_path)
        with open(ckpt_path, "rb") as f:
            return load(f)  # ModelCheckpoint(**load(f))
