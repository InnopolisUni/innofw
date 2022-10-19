import logging
from typing import Callable

from innofw.constants import UserWOKeys, DefaultS3User
from innofw.utils.s3_utils.credentials import get_s3_credentials

DEFAULT_SEQUENCE = [DefaultS3User, UserWOKeys, get_s3_credentials]


def execute_w_credentials(func: Callable):
    def executor_w_credentials(credentials=UserWOKeys, *args, **kwargs):
        output = None
        for credentials in (credentials, *DEFAULT_SEQUENCE):
            if isinstance(credentials, Callable):
                credentials = credentials()
            try:
                output = func(credentials=credentials, *args, **kwargs)
                break
            except Exception as e:
                logging.info(
                    f"could not complete the function execution. Error raised: {e}"
                )

        if output is None:
            raise ValueError(
                f"Could not complete the function with provided credentials"
            )

        return output

    return executor_w_credentials
