from typing import List, Union

from innofw.constants import Frameworks
from innofw.utils.extra import is_intersecting


# def is_intersecting(first, second):
#     f = [first] if isinstance(first, Frameworks) else first
#     s = [second] if isinstance(second, Frameworks) else second
#     return not set(f).isdisjoint(s)


def find_suitable_datamodule(task: Union[str, List[str]], framework: Frameworks):
    import inspect
    from innofw.core import datamodules

    if type(task) == str:
        task = [task]

    for t in task:
        # search for suitable datamodule from codebase
        clsmembers = inspect.getmembers(datamodules, inspect.isclass)
        for _, cls in clsmembers:
            if is_intersecting(t, cls.task) and is_intersecting(
                framework, cls.framework
            ):
                return ".".join([cls.__module__, cls.__name__])
        else:
            raise ValueError(f"Could not find data module for the {t} and {framework}")
