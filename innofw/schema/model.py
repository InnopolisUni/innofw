#
from typing import Optional, Dict, Any
import logging

#
from pydantic import root_validator, Field
import yaml
import torch

#
from .base_config import BaseConfig
from innofw.utils.find_model import find_suitable_model

# from innofw.utils.framework import map_model_to_framework


class ModelConfig(BaseConfig):
    models: Optional[Dict[str, Any]]  # todo: rename into extra or something

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        source: https://github.com/samuelcolvin/pydantic/issues/2285#issuecomment-770100339
        """
        trainer_cfg = None
        try:
            trainer_cfg = values["trainer_cfg"]
            del values["trainer_cfg"]
        except:
            pass

        all_required_field_names = {
            field.alias for field in cls.__fields__.values() if field.alias != "extra"
        }  # to support alias

        extra: Dict[str, Any] = {}
        values_copy = values.copy()
        for field_name in list(values):
            if field_name not in all_required_field_names:
                extra[field_name] = values.pop(field_name)
        values["models"] = extra
        if (
            "_target_" not in extra
            # or extra["_target_"] == "???"
            or extra["_target_"] is None
            or extra["_target_"] == ""
        ):
            model_class = find_suitable_model(values_copy["name"])
            values["models"]["_target_"] = model_class

            if trainer_cfg:
                fw = model_class.split(".")[0]
                if (
                    fw == "xgboost"
                    and "accelerator" in trainer_cfg
                    and trainer_cfg.accelerator == "gpu"
                    and torch.cuda.is_available()
                ):
                    values["models"]["tree_method"] = "gpu_hist"

                    if "gpus" in trainer_cfg:
                        import logging

                        values["models"]["gpu_id"] = (
                            trainer_cfg["gpus"][0]
                            if type(trainer_cfg["gpus"]) == list
                            else 0
                        )

                    # todo: smh handle the devices parameter

        return values

    def to_dict(self) -> dict:
        key_values = self.dict().copy()
        del key_values["models"]
        key_values.update(**self.dict()["models"])
        return key_values

    def to_yaml(self):
        key_values = self.to_dict()
        return yaml.dump(key_values)

    def save_as_yaml(self, path):
        as_dict = self.to_dict()

        with open(path, "w") as file:
            yaml.dump(as_dict, file)
