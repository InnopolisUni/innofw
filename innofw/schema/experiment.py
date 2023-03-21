# standard libraries
from typing import List
from typing import Optional
from typing import Union

import yaml
from pydantic import BaseModel

from innofw.schema.dataset import DatasetConfig
from innofw.schema.losses import Losses
from innofw.schema.model import ModelConfig

# third party libraries
# local modules


class ExperimentConfig(BaseModel):
    models: ModelConfig
    datasets: DatasetConfig

    task: str
    project: str
    # experiment_name: str

    batch_size: Optional[int]
    epochs: Optional[int]
    accelerator: str

    ckpt_path: Optional[str]
    weights_path: Optional[str]
    weights_freq: Optional[int]
    stop_param: Optional[str]
    random_seed: Optional[int]
    gpus: Optional[Union[int, List[int]]]
    devices: Optional[int]

    losses: Optional[Losses]
    # metrics: Optional[Metrics]
    # optimizers: Optional[Optimizers]
    # schedulers: Optional[Schedulers]
    # augmentations: Optional[Augmentations]
    # trainer: Optional[Trainer]
    # initializations: Optional[Initializations]
    # callbacks: Optional[Callbacks]

    def to_dict(self) -> dict:
        key_values = self.dict().copy()
        return key_values

    def to_yaml(self):
        key_values = self.to_dict()
        return yaml.dump(key_values)

    def save_as_yaml(self, path):
        as_dict = self.to_dict()

        with open(path, "w") as file:
            yaml.dump(as_dict, file)


# class InferenceConfig(MainConfig):
#     ckpt_path: str


# class TestConfig(MainConfig):
#     metrics: Metrics


# if __name__ == "__main__":
#     model = Models(
#         name="model",
#         description="something",
#         _target_="sklearn.linear_model.LinearRegression",
#     )
#     fw = Frameworks.torch
#     datasets = Datasets(
#         name="some", description="thing", task="image-segmentation", framework=fw
#     )
#     conf = InferenceConfig(model, datasets, ckpt_path="some/path")
