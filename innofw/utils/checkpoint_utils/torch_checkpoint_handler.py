import logging
from pathlib import Path
from typing import Optional
from typing import Union

import torch
from pydantic import validate_arguments

from innofw.constants import CheckpointFieldKeys
from innofw.utils import get_abs_path
from innofw.utils.checkpoint_utils.base_checkpoint_handler import (
    CheckpointHandler,
)


def get_state_dict(model):
    try:
        model_state_dict = model.state_dict()
        return model_state_dict
    except:
        pass

    try:
        model_state_dict = model[CheckpointFieldKeys.model]
        return model_state_dict
    except:
        pass

    try:
        model_state_dict = model["model_state_dict"]
        return model_state_dict
    except KeyError:
        pass

    try:
        model_state_dict = model["state_dict"]
        return model_state_dict
    except KeyError:
        pass

    # for yolov5
    try:
        model_state_dict = model["model"]
        return model
    except KeyError:
        pass

    raise ValueError("Unable to get checkpoint keys")


class TorchCheckpointHandler(CheckpointHandler):
    """
    A class that defines torch checkpoints handling

    Methods
    -------
    save_ckpt(model, dst_path: Union[str, Path], metadata: Optional[dict] = None, wrap: bool)
        saves a torch model in the destination path with given metadata
    load_ckpt(ckpt_path: Union[str, Path])
        loads a torch checkpoint from a given path
    """

    @staticmethod
    @validate_arguments
    def save_ckpt(
        model,
        dst_path: Union[str, Path],
        metadata: Optional[dict] = None,
        create_default_folder: bool = False,
        wrap=True,
    ):
        if metadata is None:
            metadata = dict()

        dst_path = Path(dst_path)
        if not dst_path.is_absolute():
            dst_path = get_abs_path(dst_path)
        if dst_path.is_dir():
            if create_default_folder and "checkpoints" not in dst_path.parts:
                dst_path /= "checkpoints"

            dst_path = dst_path / "model.ckpt"

        try:
            inner_metadata = model[CheckpointFieldKeys.metadata]
            inner_metadata.update(metadata)
            metadata = inner_metadata.copy()
        except:
            pass

        # except (TypeError, KeyError):  # catches model object is not subscriptable
        data = {
            CheckpointFieldKeys.model: get_state_dict(model),
            CheckpointFieldKeys.metadata: metadata,
        }

        if wrap:
            torch.save(data, dst_path)
        else:
            torch.save(data[CheckpointFieldKeys.model], dst_path)

        logging.info(f"Saved a checkpoint at: {dst_path}")
        return dst_path

    @staticmethod
    @validate_arguments
    def load_ckpt(
        ckpt_path: Union[str, Path]
    ) -> dict:  # check the output type
        if not Path(ckpt_path).is_absolute():
            ckpt_path = get_abs_path(ckpt_path)
        try:
            return torch.load(ckpt_path)
        except RuntimeError:
            return torch.load(ckpt_path, map_location=torch.device("cpu"))

    @validate_arguments
    def load_model(self, model, ckpt_path: Path) -> torch.nn.Module:
        """Function returns model with loaded weights"""
        if model is None:
            return get_state_dict(self.load_ckpt(str(ckpt_path)))

        model.load_state_dict(super().load_model(model, ckpt_path))
        return model
