import shutil

from omegaconf import DictConfig

from innofw.constants import Frameworks, Stages
from innofw.core.datamodules.lightning_datamodules.anomaly_detection_timeseries_dm import \
    TimeSeriesLightningDataModule
from innofw.core.models.torch.lightning_modules import (
    AnomalyDetectionTimeSeriesLightningModule
)
from innofw.utils.framework import get_datamodule
from tests.fixtures.config.datasets import anomaly_detection_timeseries_datamodule_cfg_w_target
from innofw.utils.framework import get_losses
from innofw.utils.framework import get_model
from tests.fixtures.config import losses as fixt_losses
from tests.fixtures.config import models as fixt_models
from tests.fixtures.config import optimizers as fixt_optimizers
from tests.fixtures.config import trainers as fixt_trainers
from tests.fixtures.config import schedulers as fixt_schedulers


def test_anomaly_detection():
    cfg = DictConfig(
        {
            "models": fixt_models.lstm_autoencoder_w_target,
            "trainer": fixt_trainers.trainer_cfg_w_cpu_devices,
            "losses": fixt_losses.l1_loss_w_target,
        }
    )
    model = get_model(cfg.models, cfg.trainer)
    losses = get_losses(cfg, "anomaly-detection-timeseries", Frameworks.torch)
    optimizer_cfg = DictConfig(fixt_optimizers.adam_optim_w_target)
    scheduler_cfg = DictConfig(fixt_schedulers.linear_w_target)

    module = AnomalyDetectionTimeSeriesLightningModule(
        model=model, losses=losses, optimizer_cfg=optimizer_cfg, scheduler_cfg=scheduler_cfg
    )

    assert module is not None

    datamodule: TimeSeriesLightningDataModule = get_datamodule(
        anomaly_detection_timeseries_datamodule_cfg_w_target,
        Frameworks.torch,
        task="anomaly-detection-timeseries"
    )
    datamodule.setup(Stages.train)

    module.training_step(next(iter(datamodule.train_dataloader())), 0)
    module.validation_step(next(iter(datamodule.train_dataloader())), 0)
    module.predict_step(next(iter(datamodule.train_dataloader())), 0)

    for i in range(3):
        try:
            shutil.rmtree('./tmp')
            break
        except:
            pass
