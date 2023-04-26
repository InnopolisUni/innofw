import pytest
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from innofw.constants import Frameworks
from innofw.core.models.torch.lightning_modules.segmentation import (
    SemanticSegmentationLightningModule,
)
from innofw.utils.framework import get_losses
from innofw.utils.framework import get_model
from tests.fixtures.config import losses as fixt_losses
from tests.fixtures.config import models as fixt_models
from tests.fixtures.config import optimizers as fixt_optimizers
from tests.fixtures.config import trainers as fixt_trainers


@pytest.fixture
def segmentation_module() -> LightningModule:
    cfg = DictConfig(
        {
            "models": fixt_models.deeplabv3_plus_w_target,
            "trainer": fixt_trainers.trainer_cfg_w_cpu_devices,
            "losses": fixt_losses.jaccard_loss_w_target,
        }
    )
    model = get_model(cfg.models, cfg.trainer)
    losses = get_losses(cfg, "image-segmentation", Frameworks.torch)
    optimizer_cfg = DictConfig(fixt_optimizers.adam_optim_w_target)

    module = SemanticSegmentationLightningModule(
        model=model, losses=losses, optimizer_cfg=optimizer_cfg
    )

    return module
