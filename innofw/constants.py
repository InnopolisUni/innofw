# standard libraries
from enum import Enum
from pathlib import Path
from typing import TypeVar, Optional

# third-party libraries
from pydantic import BaseModel, SecretStr


class Stages(Enum):
    train = "train"
    test = "test"
    predict = "predict"


class Frameworks(Enum):
    torch = "torch"
    sklearn = "sklearn"
    xgboost = "xgboost"
    catboost = "catboost"
    ultralytics = "ultralytics"
    mmdetection = "mmdetection"
    none = "none"

PathLike = TypeVar(
    "PathLike", str, Path
)  # ref: https://stackoverflow.com/questions/58541722/what-is-the-correct-way-in-python-to-annotate-a-path-with-type-hints


class BucketNames(Enum):
    model_zoo = "pretrained"
    data_mart = "public-datasets"


class DefaultFolders(Enum):
    remote_model_weights_save_dir = Path("pretrained")


class S3Credentials(BaseModel):
    ACCESS_KEY: Optional[SecretStr] = None
    SECRET_KEY: Optional[SecretStr] = None


class Status(Enum):
    FAIL = 0
    SUCCESS = 1


class S3FileTags(Enum):
    hash_value = "hash_value"


class ModelType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"


class CheckpointFieldKeys(Enum):
    model = "model"
    metadata = "metadata"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class CLI_FLAGS(Enum):
    DISABLE = "NO_CLI"
    ENABLE = "WITH_CLI"


# ==== instances ====
"""
s3 user with following rights to two buckets: data_mart and model_zoo(ref. BucketNames):
- s3:GetObjectTagging
- s3:GetObject
- s3:GetBucketLocation
"""
DefaultS3User = S3Credentials(
    ACCESS_KEY=SecretStr("MK2IJPXYGYX1ZINEJKLJ"),
    SECRET_KEY=SecretStr("mLBjBz8aClRjR6j+79v7jMVUQZo1k8rwPyntGWxY"),
)

UserWOKeys = S3Credentials(
    ACCESS_KEY=None,
    SECRET_KEY=None,
)

DEFAULT_STORAGE_URL: str = "https://api.blackhole.ai.innopolis.university:443"

# === semantic segmentation ===
# constants
class SegDataKeys(Enum):
    image = 'image'
    label = 'label'
    filename = 'name'
    coords = 'coords'
    metadata = 'metadata'


class SegOutKeys(Enum):  # todo: use it somehow
    predictions = "predictions"
    metrics = "metrics"