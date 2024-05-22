from typing import Any
from innoframework.schema.model_metadata import ModelMetadata
from pydantic import BaseModel

class ModelCheckpoint(BaseModel):
    model: Any
    metadata: ModelMetadata
