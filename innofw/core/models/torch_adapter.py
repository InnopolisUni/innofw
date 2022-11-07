import logging
import os

# third party libraries
import hydra.utils
import pytorch_lightning as pl
import torch

# local modules
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from .base import BaseModelAdapter
from innofw.constants import Stages
from innofw.core.models import register_models_adapter
from innofw.utils.defaults import get_default
from innofw.utils.checkpoint_utils import TorchCheckpointHandler


class ModelCheckpointWithLogging(ModelCheckpoint):
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        logging.info(f"Saved a checkpoint at: {filepath}")


@register_models_adapter(name="torch_adapter")
class TorchAdapter(BaseModelAdapter):
    """
    Adapter for working with CatBoost models
    ...

    Attributes
    ----------
    callbacks : list
        list of callbacks to perform
    trainer : pl.Trainer
        PyTorch lighting trainer to perform training
    Methods
    -------
    set_stop_params(stop_param)
      configure parameters for early stopping
    predict(x):
        returns result of prediction and saves them

    """
    def _test(self, data):
        pass

    @staticmethod
    def is_suitable_model(model) -> bool:
        return isinstance(model, torch.nn.Module)

    def _train(self, data):
        pass

    def __init__(
        self,
        model,
        task,
        log_dir,
        losses=None,
        optimizers_cfg=None,
        schedulers_cfg=None,
        callbacks=None,
        initializations=None,
        trainer_cfg=None,
        weights_path=None,
        weights_freq=None,
        stop_param=None,
        project=None,
        experiment=None,
        *args,
        **kwargs,
    ):
        super().__init__(model, log_dir, TorchCheckpointHandler())
        self.callbacks = callbacks or []

        self.set_checkpoint_save(weights_path, weights_freq, project, experiment)
        if stop_param:
            self.set_stop_params(stop_param)
        self.ckpt_path = None

        # initialize model weights with function

        if initializations is not None:
            logging.info("initializing the model")
            initializations.init_weights(model)

        objects = {
            "lightning_module": None,
            "losses": losses,
            "optimizers_cfg": optimizers_cfg,
            "schedulers_cfg": schedulers_cfg,
            "callbacks": callbacks,
            "trainer_cfg": trainer_cfg,
        }
        framework = "torch"
        for key, value in objects.items():
            if not value:
                objects[key] = get_default(key, framework, task)

        self.pl_module = objects["lightning_module"](
            model,
            objects["losses"],
            objects["optimizers_cfg"],
            objects["schedulers_cfg"],
        )
        #

        if callable(objects["trainer_cfg"]):
            self.trainer = objects["trainer_cfg"](
                callbacks=self.callbacks,
                default_root_dir=self.log_dir,
                check_val_every_n_epoch=1,
            )
        elif "_target_" in objects["trainer_cfg"]:
            self.trainer = hydra.utils.instantiate(
                objects["trainer_cfg"],
                callbacks=self.callbacks,
                default_root_dir=self.log_dir,
                check_val_every_n_epoch=1,
            )
        else:
            self.trainer = pl.Trainer(
                **trainer_cfg,
                callbacks=self.callbacks,
                default_root_dir=self.log_dir,
                check_val_every_n_epoch=1,
            )

    # def resume_checkpoint(self, ckpt_path):
    #     self.ckpt_path = ckpt_path

    def _predict(self, x):
        x.setup(Stages.predict)
        return self.trainer.predict(self.pl_module, x)

    def predict(self, datamodule, ckpt_path=None):
        if ckpt_path is not None:
            self.pl_module = self.ckpt_handler.load_model(self.pl_module, ckpt_path)
            result = self._predict(datamodule)
            return datamodule.save_preds(
                result, stage=Stages.predict, dst_path=self.log_dir
            )

        # result = self.trainer.predict(self.pl_module, datamodule)  # , ckpt_path=ckpt_path
        # return datamodule.save_preds(
        #     result, stage=Stages.predict, dst_path=self.log_dir
        # )

    def train(self, data_module, ckpt_path=None):
        self.trainer.fit(self.pl_module, data_module, ckpt_path=ckpt_path)

    def test(self, data_module):
        outputs = self.trainer.test(self.pl_module, data_module)
        return outputs  # [0]["metrics"]

    def set_stop_params(self, stop_param):
        self.callbacks.append(EarlyStopping(monitor="val_loss", patience=stop_param))

    def set_checkpoint_save(self, weights_path, weights_freq, project, experiment):
        if weights_path:
            self.callbacks.append(
                ModelCheckpointWithLogging(
                    dirpath=weights_path,
                    filename=f"{project}_{experiment}" + "_{epoch}",
                    every_n_epochs=weights_freq,
                    save_top_k=-1,
                )
            )
        else:
            log_dir = (
                self.log_dir
                if self.log_dir.name == "checkpoints"
                else self.log_dir / "checkpoints"
            )

            self.callbacks.append(
                ModelCheckpointWithLogging(
                    dirpath=log_dir,
                    filename=f"model",
                    every_n_epochs=weights_freq,
                    save_top_k=-1,
                )
            )
