from ..base_adapter import BaseAdapter


class UltralyticsSchedulerBaseAdapter(BaseAdapter):
    """Class defines adapter interface to conform to Ultralytics scheduler specifications

        Methods
        -------
        adapt(scheduler: DictConfig) -> dict
            converts the scheduler configuration into Ultralytics suitable format
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
