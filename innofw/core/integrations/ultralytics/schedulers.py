from ..base_adapter import BaseAdapter


class YOLOV5SchedulerBaseAdapter(BaseAdapter):
    def __init__(self):
        self.opt = {}
        self.hyp = {
            "lr0": 1e-2,
            "lrf": 1e-1,
        }

    def adapt(self, scheduler) -> dict:
        return {"opt": self.opt, "hyp": self.hyp}

    def from_cfg(self, cfg):
        return {}, {}

    def from_obj(self, obj):
        return {}, {}
