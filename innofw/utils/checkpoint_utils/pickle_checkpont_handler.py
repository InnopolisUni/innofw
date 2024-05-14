import logging
from pathlib import Path
from pickle import dump
from pickle import load
from typing import Optional
from typing import Union

import sklearn

from innofw.constants import CheckpointFieldKeys
from innofw.utils import get_abs_path
from innofw.utils.checkpoint_utils.base_checkpoint_handler import (
    CheckpointHandler,
)

# from innoframework.schema.model_checkpoint import ModelCheckpoint


class PickleCheckpointHandler(CheckpointHandler):
    """
    A class that defines .pickle checkpoints handling

    Methods
    -------
    save_ckpt(model, dst_path: Union[str, Path], metadata: Optional[dict] = None, wrap: bool)
        saves a pickle model in the destination path with given metadata
    load_ckpt(ckpt_path: Union[str, Path])
        loads a pickle checkpoint from a given path
    """

    @staticmethod
    def save_ckpt(
        model: sklearn.base.BaseEstimator,
        dst_path: Union[str, Path],
        metadata: Optional[dict] = None,
        file_extension: str = ".pickle",
        create_default_folder: bool = False,
        **kwargs
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
