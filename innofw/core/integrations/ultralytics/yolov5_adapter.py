# standard libraries
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml
from yolov5 import detect as yolov5_detect
from yolov5 import train as yolov5_train
from yolov5 import val as yolov5_val

from ..base_integration_models import BaseIntegrationModel
from .datamodule import UltralyticsDataModuleAdapter
from .losses import UltralyticsLossesBaseAdapter
from .optimizers import UltralyticsOptimizerBaseAdapter
from .schedulers import UltralyticsSchedulerBaseAdapter
from .trainer import UltralyticsTrainerBaseAdapter
from innofw.constants import Frameworks
from innofw.core.models import BaseModelAdapter
from innofw.core.models import register_models_adapter
from innofw.utils.checkpoint_utils import TorchCheckpointHandler

# third party libraries
# local modules

YOLOV5_VALID_ARCHS = ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]


class YOLOv5Model(BaseIntegrationModel):
    """Class defines adapter interface to conform to YOLOv5 model specifications

    Attributes
    ----------
    framework: Frameworks
        framework through which the model is implemented
    """

    framework = Frameworks.torch

    def __init__(self, arch, *args, **kwargs):
        self.cfg = arch
        assert (
            arch in YOLOV5_VALID_ARCHS
        ), f"arch should one of following: {YOLOV5_VALID_ARCHS}"


@register_models_adapter(name="yolov5_adapter")
class YOLOV5Adapter(BaseModelAdapter):
    """
    Adapter for working with Yolov5 models
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
    train(data: YOLOV5DataModuleAdapter, ckpt_path=None):
        trains the model
    predict(x):
        returns result of prediction and saves them

    """

    @staticmethod
    def is_suitable_model(model) -> bool:
        return isinstance(model, YOLOv5Model)

    def _test(self, data):
        pass

    def _train(self, data):
        pass

    def _predict(self, data):
        pass

    framework = Frameworks.torch

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
            "upload_dataset": False,
            "mmdet_tags": False,
            "quad": False,
            "exist_ok": True,
            "multi_scale": False,
            "sync_bn": False,
            "cos_lr": False,
            "image_weights": False,
            "noplots": False,
            "noautoanchor": False,
            "noval": False,
            "nosave": False,
            "rect": False,
            "single_cls": False,
            "freeze": [0],
            "entity": None,
            "seed": 42,
            "s3_upload_dir": None,
            "neptune_project": None,
            "neptune_token": None,
            "artifact_alias": "latest",
            "bbox_interval": -1,
            "local_rank": -1,
            "label_smoothing": 0.0,
            "cache": "ram",
            "image_weights": False,
            "workers": 8,
        }

        name = "save-period"

        if weights_freq is not None:
            self.opt[name] = weights_freq
        else:
            self.opt[name] = -1

        if weights_path is not None:
            self.opt["save_dir"] = weights_path
        else:
            self.opt["save_dir"] = str(self.log_dir)

        self.hyp = {
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 0.05,
            "cls": 0.5,
            "cls_pw": 1.0,
            "obj": 1.0,
            "obj_pw": 1.0,
            "iou_t": 0.2,
            "anchor_t": 4.0,
            "fl_gamma": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
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
        # try:
        #     self.batch_size = trainer_cfg.batch_size
        # except:
        #     self.batch_size = 4

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

        if ckpt_path is None:
            weights = str(self.model.cfg) + ".pt"

            self.opt.update(
                imgsz=data.imgsz,
                data=data.data,
                workers=data.workers,
                batch_size=data.batch_size,
                cfg=str(self.model.cfg) + ".yaml",
                weights=weights,
                device=self.device,
                hf_model_id=None,
                project="something",
                epochs=self.epochs,
            )
            self._yolov5_train.run(
                hyp="hyp.yaml",
                **self.opt,
            )
        else:
            try:
                del self.opt["weights"]
            except:
                pass

            ckpt_path = TorchCheckpointHandler().convert_to_regular_ckpt(
                ckpt_path, inplace=False, dst_path=None
            )
            ckpt = torch.load(ckpt_path)
            # inner_opt = self.opt.copy()
            # del inner_opt['resume']
            # ckpt['opt'] = inner_opt
            # torch.save(ckpt, ckpt_path)

            if ckpt["epoch"] + 1 <= self.epochs:
                self.opt["epochs"] = ckpt["epoch"] + 2
            else:
                self.opt["epochs"] = self.epochs

            try:
                self.opt["save_period"] = self.opt["save-period"]
                del self.opt["save-period"]
            except:
                pass

            self.opt.update(
                imgsz=data.imgsz,
                data=data.data,
                workers=data.workers,
                batch_size=data.batch_size,
                cfg=str(self.model.cfg) + ".yaml",
                resume=str(ckpt_path),
                device=self.device,
                evolve=False,
                hf_model_id=None,
                project="something",
            )

            # create opt.yaml
            opt_file = ckpt_path.parent.parent / "opt.yaml"
            with open(opt_file, "w+") as f:
                # inside the opt.yaml include key `hyp` with dicts
                self.opt["hyp"] = self.hyp
                yaml.dump(self.opt, f)

            self._yolov5_train.run(
                resume=str(ckpt_path),
            )
            opt_file.unlink(missing_ok=True)

        self.update_checkpoints_path()

    @property
    def _yolov5_train(self):
        return yolov5_train

    @property
    def _yolov5_val(self):
        return yolov5_val

    @property
    def _yolov5_predict(self):
        return yolov5_detect

    def predict(self, data: UltralyticsDataModuleAdapter, ckpt_path=None):
        data.setup()

        ckpt_path = TorchCheckpointHandler().convert_to_regular_ckpt(
            ckpt_path, inplace=False, dst_path=None
        )

        params = dict(
            weights=ckpt_path,
            imgsz=data.imgsz,
            conf_thres=0.25,
            iou_thres=0.45,
            device=self.device,
            augment=False,
            project=self.opt["project"],
            name=self.opt["name"],
        )

        if (
            str(data.infer_source).startswith("rts")
            or Path(data.infer_source).is_file()
        ):
            params.update(source=data.infer_source)
        else:
            params.update(source=Path(data.infer_source) / "images", data=data.data)

        params.update(exist_ok=True)
        self._yolov5_predict.run(**params)

        self.update_checkpoints_path()

    def test(self, data: UltralyticsDataModuleAdapter, ckpt_path=None):
        data.setup()

        ckpt_path = TorchCheckpointHandler().convert_to_regular_ckpt(
            ckpt_path, inplace=False, dst_path=None
        )

        self._yolov5_val.run(
            imgsz=data.imgsz,
            data=data.data,
            workers=data.workers,
            batch_size=data.batch_size,
            device=self.device,
            weights=ckpt_path,
            task="test",
        )

        self.update_checkpoints_path()
