#
from pydantic import BaseModel
from typing import Any

#
from innoframework.schema.model_metadata import ModelMetadata


class ModelCheckpoint(BaseModel):
    model: Any
    metadata: ModelMetadata
