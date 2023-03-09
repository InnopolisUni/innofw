from pydantic import BaseModel


# @dataclass
class BaseConfig(BaseModel):
    """Class having required fields"""

    # required fields
    name: str
    description: str
