# standard libraries
import logging
import os
import yaml
from pathlib import Path
from typing import Optional

# third party libraries
import torch

# local modules
from innofw.constants import Frameworks
from innofw.utils.checkpoint_utils import TorchCheckpointHandler
from .trainer import UltralyticsTrainerBaseAdapter
from .losses import UltralyticsLossesBaseAdapter
from .optimizers import UltralyticsOptimizerBaseAdapter
from .datamodule import UltralyticsDataModuleAdapter
from .schedulers import UltralyticsSchedulerBaseAdapter
from innofw.core.models import BaseModelAdapter, register_models_adapter
from ultralytics import YOLO


@register_models_adapter(name="ultralytics_adapter")
class UltralyticsAdapter(BaseModelAdapter):
    """
    Adapter for working with Ultralytics models
    ...

    Attributes
    ----------
    device
        device for model training
    epochs : int
        maximum number of epochs
    log_dir : Path
        path to save logs
    Methods
    -------
    train(data: UltralyticsDataModuleAdapter, ckpt_path=None):
        trains the model
    predict(x):
        returns result of prediction and saves them

    """

    @staticmethod
    def is_suitable_model(model) -> bool:
        return isinstance(model, YOLO)

    def _test(self, data):
        pass

    def _train(self, data):
        pass

    def _predict(self, data):
        pass

    framework = Frameworks.ultralytics

    def __init__(
        self,
        model,
        log_dir,
        trainer_cfg,
        augmentations=None,
        optimizers_cfg=None,
        schedulers_cfg=None,
        losses=None,
        callbacks=None,
        stop_param=None,
        weights_path=None,
        weights_freq: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(model, log_dir)

        trainer = UltralyticsTrainerBaseAdapter().adapt(trainer_cfg)
        self.device, self.epochs = trainer["device"], trainer["epochs"]
        self.log_dir = Path(log_dir)

        self.opt = {
            "project": str(self.log_dir.parents[0]),
            "name": self.log_dir.name,
            "patience": stop_param,
            "label_smoothing": 0.0,
            "cache": "ram",
            "workers": 8,
            "seed": 42,
            "rect": False,
            "single_cls": False,
            "cos_lr": False,
            "exist_ok": True,
        }

        self.hyp = {
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 0.05,
            "cls": 0.5,
        }

        optimizers = UltralyticsOptimizerBaseAdapter().adapt(optimizers_cfg)
        self.opt = {**self.opt, **optimizers["opt"]}
        self.hyp = {**self.hyp, **optimizers["hyp"]}

        schedulers = UltralyticsSchedulerBaseAdapter().adapt(schedulers_cfg)
        self.opt = {**self.opt, **schedulers["opt"]}
        self.hyp = {**self.hyp, **schedulers["hyp"]}

        losses = UltralyticsLossesBaseAdapter().adapt(losses)
        self.opt = {**self.opt, **losses["opt"]}
        self.hyp = {**self.hyp, **losses["hyp"]}

        if schedulers_cfg is not None:
            if "lr0" in schedulers_cfg:
                self.hyp.update(
                    lr0=schedulers_cfg.lr0,
                )
            if "lrf" in schedulers_cfg:
                self.hyp.update(
                    lr0=schedulers_cfg.lrf,
                )
        if optimizers_cfg is not None:
            self.opt.update(optimizer=optimizers_cfg._target_.split(".")[-1])
        with open("hyp.yaml", "w+") as f:
            yaml.dump(self.hyp, f)

    def update_checkpoints_path(self):
        try:
            (self.log_dir / "weights").rename(self.log_dir / "checkpoints")

            try:
                dst_path = list((self.log_dir / "checkpoints").iterdir())[0]
                logging.info(f"Saved a checkpoint at: {dst_path}")
            except:
                pass
        except Exception as e:
            pass
            # print(e)
            # logging.info(f"{e}")

    def train(self, data: UltralyticsDataModuleAdapter, ckpt_path=None):
        data.setup()
        name = str(self.log_dir).replace(str(self.log_dir.parents[2]) + os.sep, "")
        self.opt.update(
            project="train",
            name=name,)

        if ckpt_path is not None:
            try:
                ckpt_path = TorchCheckpointHandler().convert_to_regular_ckpt(
                    ckpt_path, inplace=False, dst_path=None, set_epoch=0
                )
                self.opt.update(resume=str(ckpt_path))
                self.model.ckpt["epoch"] = 0
                self.model.ckpt_path = ckpt_path
            except Exception as e:
                print(e)

        self.opt.update(
            device=self.device,
            epochs=self.epochs,
            imgsz=data.imgsz,
            data=data.data,
            workers=data.workers,
            batch=data.batch_size,
        )
        self.model.train(**self.opt, **self.hyp)

        self.update_checkpoints_path()

    def predict(self, data: UltralyticsDataModuleAdapter, ckpt_path=None):
        data.setup()

        if ckpt_path:
            ckpt_path = TorchCheckpointHandler().convert_to_regular_ckpt(
                ckpt_path, inplace=False, dst_path=None
            )

            self.model._load(str(ckpt_path))

        params = dict(
            conf=0.25,
            iou=0.45,
            save=True,
            device=self.device,
            augment=False,
            project=str(self.log_dir.parent),
            name=str(self.log_dir.name),
        )
        params.update(source=str(data.infer_source))
        params.update(exist_ok=True)
        self.model.predict(**params)
        self.update_checkpoints_path()

    def test(self, data: UltralyticsDataModuleAdapter, ckpt_path=None):
        data.setup()

        ckpt_path = TorchCheckpointHandler().convert_to_regular_ckpt(
            ckpt_path, inplace=False, dst_path=None
        )
        self.model._load(ckpt_path)
        self.model.val(
            imgsz=data.imgsz,
            data=data.data,
            iou=0.6,
            batch=data.batch_size,
            device=self.device,
        )

        self.update_checkpoints_path()
