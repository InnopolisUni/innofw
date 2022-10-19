import os
import sys
from pathlib import Path
from typing import Union, Optional

from innofw.constants import CLI_FLAGS


def find_file_by_ext(
    path: Union[str, Path], ext=".csv"
) -> Optional[Path]:  # todo: write a test
    path = Path(path)

    if path is None:
        return
    if str(path).endswith(ext):
        return path
    target_files = list(path.rglob(f"*{ext}"))
    if len(target_files) == 0:
        raise ValueError(f"Unable to find file with extension: {ext}")
    else:
        return target_files[0]


# source: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
def query_yes_no(question, default="yes") -> bool:
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    try:
        if os.environ[CLI_FLAGS.DISABLE.value] == "True":
            return valid["n"]
    except:
        pass

    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def find_path(path):
    # get first directory in the path
    dir = next(os.walk(path))[1][0]
    return os.path.join(path, dir)
