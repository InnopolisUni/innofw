# standard libraries
# third party libraries
from pydantic import FilePath
from pydantic import validate_arguments

from innofw.schema.model_metadata import ModelMetadata
from innofw.utils.checkpoint_utils.base_checkpoint_handler import (
    CheckpointHandler,
)
from innofw.utils.checkpoint_utils.pickle_checkpont_handler import (
    PickleCheckpointHandler,
)
from innofw.utils.checkpoint_utils.torch_checkpoint_handler import (
    TorchCheckpointHandler,
)

#


@validate_arguments
def add_metadata2model(ckpt_path: FilePath, metadata: dict, check_schema=True):
    if check_schema:
        metadata = ModelMetadata(**metadata).dict(
            by_alias=True
        )  # dict(ModelMetadata(**metadata))

    if ckpt_path.suffix in [".ckpt", ".pt"]:
        TorchCheckpointHandler().add_metadata(ckpt_path, metadata=metadata)
    elif ckpt_path.suffix in [".pickle", ".pkl", ".cmb"]:
        PickleCheckpointHandler().add_metadata(ckpt_path, metadata=metadata)
    else:
        raise NotImplementedError(
            f"unable to process extension: {ckpt_path.suffix}"
        )


@validate_arguments
def load_metadata(file_path: FilePath):
    if file_path.suffix in [".ckpt", ".pt"]:
        metadata = TorchCheckpointHandler().load_metadata(file_path)
    elif file_path.suffix in [
        ".pickle",
        ".pkl",
        ".cmb",
    ]:
        metadata = PickleCheckpointHandler().load_metadata(file_path)
    else:
        raise NotImplementedError(
            f"unable to process extension: {file_path.suffix}"
        )

    return metadata
