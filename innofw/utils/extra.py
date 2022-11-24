import logging
from typing import Callable

DOWNLOAD_ATTEMPTS = 3


def execute_with_retries(func: Callable):
    """Executes the command multiple times until success"""

    def executor_with_retries(*args, **kwargs):
        output = None

        for i in range(DOWNLOAD_ATTEMPTS):
            try:
                output = func(*args, **kwargs)
                break
            except Exception as e:
                logging.info(
                    f"could not complete the function execution. Error raised: {e}"
                )

        if output is None:
            raise ValueError(
                f"Could not complete the function in {DOWNLOAD_ATTEMPTS} attempts"
            )

        return output

    return executor_with_retries


def is_intersecting(first, second):
    f = first if isinstance(first, list) else [first]
    s = second if isinstance(second, list) else [second]
    return not set(f).isdisjoint(s)
