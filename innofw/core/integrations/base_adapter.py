from abc import ABC
from abc import abstractmethod


class BaseAdapter(ABC):
    """An abstract class that defines interface for adapters

    Methods
    -------
    adapt(*args, **kwargs)
        formats input
    from_cfg(cfg):
        formats contents of the config file
    """

    @abstractmethod
    def adapt(self, *args, **kwargs):
        pass

    # @staticmethod
    @abstractmethod
    def from_cfg(self, cfg):
        pass

    # @staticmethod
    @abstractmethod
    def from_obj(self, obj):
        pass
