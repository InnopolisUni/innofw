#
from typing import Optional, Dict, Any, Union

import yaml
from pathlib import Path

#
from omegaconf.listconfig import ListConfig
from pydantic import BaseModel, root_validator, Field

#
from innofw.schema.base_config import BaseConfig
from innofw.utils.find_datamodule import find_suitable_datamodule

# todo: refactor this
"""
    Why?
        because current implementation does not separate required and optional fields
"""


class DatasetConfig(BaseConfig):
    task: Union[ListConfig, list, str]
    markup_info: str
    date_time: str
    # _target_: Optional[str] = Field(alias='_target_')

    datasets: Optional[Dict[str, Any]]  # todo: rename into extra or something

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        source: https://github.com/samuelcolvin/pydantic/issues/2285#issuecomment-770100339
        """
        all_required_field_names = {
            field.alias
            for field in cls.__fields__.values()
            if field.alias != "datasets"
        }  # to support alias

        extra: Dict[str, Any] = {}
        values_copy = values.copy()
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name == "framework":
                    continue
                extra[field_name] = values.pop(field_name)
        values["datasets"] = extra

        if (  # todo: write thorough tests for each case
            "_target_" not in extra
            # or extra["_target_"] == "???"
            or extra["_target_"] is None
            or extra["_target_"] == ""
        ):
            dataset_class = find_suitable_datamodule(
                values_copy["task"], values["framework"]
            )  # , extra
            values["datasets"]["_target_"] = dataset_class
        return values

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> dict:
        key_values = self.dict().copy()
        del key_values["datasets"]
        key_values.update(**self.dict()["datasets"])
        return key_values

    def to_yaml(self):
        key_values = self.to_dict()
        return yaml.dump(key_values)

    def save_as_yaml(self, path: Path):
        # if path.is_file():
        path.parent.mkdir(exist_ok=True, parents=True)
        as_dict = self.to_dict()

        with open(path, "w+") as file:
            yaml.dump(as_dict, file)
