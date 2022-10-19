from pydantic import BaseModel, conint, validator
from pydantic.dataclasses import dataclass


# @dataclass
class BaseConfig(BaseModel):
    """Class having required fields"""

    # required fields
    name: str
    description: str
