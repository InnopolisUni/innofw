# local modules
from ..base_adapter import BaseAdapter


class YOLOV5OptimizerBaseAdapter(BaseAdapter):
    def __init__(self):
        self.possible_values = {"optimizer": ["Adam", "AdamW", "SGD"]}
        self.opt = {"optimizer": "Adam"}
        self.hyp = {"momentum": 0.937, "weight decay": 5e-4}

    def adapt(self, optimizer) -> dict:
        return {"opt": self.opt, "hyp": self.hyp}

    def from_cfg(self, cfg):
        return {}, {}

    def from_obj(self, obj):
        return {}, {}
