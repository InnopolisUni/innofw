import pytest
from omegaconf import DictConfig

# local modules
from innofw.constants import Frameworks
from innofw.utils import get_project_root
from innofw.utils.config import read_cfg_2_dict
from innofw.utils.framework import get_losses


segmentation_loss_cfgs = (
    get_project_root() / "config" / "losses" / "semantic-segmentation"
).iterdir()
segmentation_loss_cfgs = [[i] for i in segmentation_loss_cfgs]


@pytest.mark.parametrize(["loss_cfg_file"], segmentation_loss_cfgs)
def test_creation(loss_cfg_file):
    task = "image-segmentation"
    framework = Frameworks.torch
    loss_cfg = read_cfg_2_dict(loss_cfg_file)

    get_losses(DictConfig({"losses": loss_cfg}), task, framework)
