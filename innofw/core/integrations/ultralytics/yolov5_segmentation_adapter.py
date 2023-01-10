from .yolov5_adapter import YOLOV5Adapter

from yolov5.segment import (
    train as yolov5_train,
    predict as yolov5_predict,
    val as yolov5_val,
)
from ..base_integration_models import BaseIntegrationModel
from innofw.constants import Frameworks
from innofw.core.models import register_models_adapter


YOLOV5_VALID_ARCHS = [
    "yolov5s-seg",
    "yolov5m-seg",
    "yolov5l-seg",
    "yolov5x-seg",
]


class YOLOv5SegModel(BaseIntegrationModel):
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


@register_models_adapter(name="yolov5_seg_adapter")
class YOLOV5SegAdapter(YOLOV5Adapter):
    @staticmethod
    def is_suitable_model(model) -> bool:
        return isinstance(model, YOLOv5SegModel)

    @property
    def _yolov5_train(self):
        return yolov5_train

    @property
    def _yolov5_val(self):
        return yolov5_val

    @property
    def _yolov5_predict(self):
        return yolov5_predict
