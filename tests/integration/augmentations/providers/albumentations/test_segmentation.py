from omegaconf import DictConfig

from innofw.constants import Frameworks
from innofw.utils.framework import get_obj

# local


def test_augmentation_creation():
    cfg = DictConfig(
        {
            "augmentations": {
                "task": ["all"],
                "implementations": {
                    "torch": {
                        "Compose": {
                            "object": {
                                "_target_": "albumentations.Compose",
                                "transforms": [
                                    {
                                        "_target_": "albumentations.Transpose",
                                        "p": 0.5,
                                        "always_apply": False,
                                    },
                                    {"_target_": "albumentations.Flip"},
                                    {
                                        "_target_": "albumentations.RandomRotate90"
                                    },
                                    {
                                        "_target_": "albumentations.ElasticTransform",
                                        "p": 0.3,
                                    },
                                ],
                            }
                        }
                    },
                },
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.torch
    augmentations = get_obj(cfg, "augmentations", task, framework)

    assert augmentations is not None
