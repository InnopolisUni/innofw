from abc import abstractmethod, ABC


class BaseAdapter(ABC):
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
