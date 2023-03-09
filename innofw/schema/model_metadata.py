from typing import Dict

from pydantic import BaseModel
from pydantic import Field


class ModelMetadata(BaseModel):
    target: str = Field(..., alias="_target_")  # path to the model class
    name: str
    weights: str  # path to the model weights
    data: str  # path to the dataset on which model has been trained
    description: str  # model description
    metrics: Dict[str, float]  # model score on testing data
