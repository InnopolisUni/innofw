import shutil
import tempfile
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Optional
from typing import Union

from pydantic import FilePath
from pydantic import validate_arguments

from innofw.constants import CheckpointFieldKeys
from innofw.utils import get_abs_path


class CheckpointHandler(ABC):
    """
    An abstract class that defines common interface for checkpoint handling

    Methods
    -------
    save_ckpt(model, dst_path: Union[str, Path], metadata: Optional[dict] = None, wrap: bool)
        saves a model in the destination path with given metadata
    load_ckpt(ckpt_path: Union[str, Path])
        loads a checkpoint from a given path
    """

    @staticmethod
    @abstractmethod
    def save_ckpt(
        model,
        dst_path: Union[str, Path],
        metadata: Optional[dict] = None,
        wrap: bool = True,
    ) -> Path:
        pass

    @staticmethod
    @abstractmethod
    def load_ckpt(ckpt_path: Union[str, Path]):
        pass

    @validate_arguments
    def add_metadata(self, file_path: FilePath, metadata: dict):
        """Function adds metadata to the existing file with model checkpoint"""
        # load the checkpoint
        if not file_path.is_absolute():
            file_path = get_abs_path(file_path)

        model = self.load_ckpt(file_path)
        # create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # save the model with metadata there
            new_file_path = Path(tmpdirname, file_path.name)
            self.save_ckpt(model, new_file_path, metadata)

            # delete original file
            file_path.unlink()
            # move file from temp directory to original file path
            shutil.move(new_file_path, file_path)

    @validate_arguments
    def load_metadata(self, file_path: Path) -> dict:
        if not file_path.is_absolute():
            file_path = get_abs_path(file_path)
        return self.load_ckpt(file_path)[CheckpointFieldKeys.metadata]

    @validate_arguments
    def load_model(
        self, model, ckpt_path: Path  # make model parameter optional
    ):
        ckpt_path = get_abs_path(ckpt_path)
        try:
            ckpt = self.load_ckpt(ckpt_path)[CheckpointFieldKeys.model]
        except Exception as e:
            ckpt = self.load_ckpt(ckpt_path)
        if isinstance(ckpt, Iterable) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        return ckpt

    @validate_arguments
    def convert_to_regular_ckpt(
        self,
        ckpt_path: Path,
        dst_path: Optional[Path] = None,
        inplace: bool = True,
        set_epoch: int = -1
    ) -> Path:
        model = self.load_model(None, ckpt_path)
        if set_epoch != -1:
            if "epoch" in model.keys():
                model["epoch"] = set_epoch
        if inplace:
            tmp_path = Path(tempfile.mkdtemp()) / ckpt_path.name
            self.save_ckpt(model, tmp_path, None, wrap=False)
            shutil.move(tmp_path, ckpt_path)
            return ckpt_path
        elif dst_path is not None:
            if not dst_path.is_absolute():
                dst_path = get_abs_path(dst_path)

            if dst_path.suffix == "":
                dst_path /= ckpt_path.name

            dst_path.parent.mkdir(exist_ok=True, parents=True)

            self.save_ckpt(model, dst_path, None, wrap=False)
            return dst_path
        else:
            tmp_path = Path(tempfile.mkdtemp()) / ckpt_path.name
            self.save_ckpt(model, tmp_path, None, wrap=False)
            return tmp_path
