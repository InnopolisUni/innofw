from ..base_adapter import BaseAdapter


class UltralyticsLossesBaseAdapter(BaseAdapter):
    """Class defines adapter interface to conform to Ultralytics loss specifications

        Methods
        -------
        adapt(loss: DictConfig) -> dict
            converts the loss configuration into Ultralytics suitable format
    """
    def __init__(self):
        self.opt = {}
        self.hyp = {
            "box": 0.05,
            "cls": 0.5,
            # "cls_pw": 1.0,
            # "obj": 1.0,
            # "obj_pw": 1.0,
            # "fl_gamma": 0.0,
        }

    def adapt(self, losses) -> dict:
        return {"opt": self.opt, "hyp": self.hyp}

    def from_cfg(self, cfg):
        return {}, {}

    def from_obj(self, obj):
        return {}, {}
