from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Base class for custom metric creation
    ...

    Attributes
    ----------
    self.metric : metric
        metric function to perform
    Methods
    -------
    """
    def __init__(self):
        if self.metric is None:
            raise ValueError("self.metric attribute should be specified")

    def __call__(self, *args, **kwargs):
        return self.metric(*args, **kwargs)


__METRIC_DICT__ = dict()


def register_custom_metrics(name, task, framework, description):
    def register_function_fn(cls):
        if name in __METRIC_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, BaseMetric):
            raise ValueError("Class %s is not a subclass of %s" % (cls, BaseMetric))
        __METRIC_DICT__[name] = {
            "task": task,
            "framework": framework,
            "description": description,
        }
        init = cls()
        init.__name__ = name
        init.description = description
        return init

    return register_function_fn
