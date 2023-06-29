from ..base_adapter import BaseAdapter


class YOLOV5SchedulerBaseAdapter(BaseAdapter):
    """Class defines adapter interface to conform to YOLOv5 scheduler specifications

    Methods
    -------
    adapt(scheduler: DictConfig) -> dict
        converts the scheduler configuration into YOLOv5 suitable format
    """

    def __init__(self):
        self.opt = {}
        self.hyp = {}

    def adapt(self, scheduler) -> dict:
        return {"opt": self.opt, "hyp": self.hyp}

    def from_cfg(self, cfg):
        return {}, {}

    def from_obj(self, obj):
        return {}, {}
