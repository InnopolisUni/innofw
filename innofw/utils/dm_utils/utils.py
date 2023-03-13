import os
import sys
from pathlib import Path
from typing import Union, Optional

from innofw.constants import CLI_FLAGS

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

def find_folder_with_images(
    path: Union[str, Path]
) -> Optional[Path]:
    try:
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Invalid path: {path}")
        if not path.is_dir():
            raise ValueError(f"Invalid directory path: {path}")
        target_folders = list(filter(lambda f: f.is_dir() and any(f.glob('*.jpg')) or any(f.glob('*.png')), path.iterdir()))
        if len(target_folders) == 0:
            raise ValueError(f"No folders found with images in {path}")
        elif len(target_folders) > 1:
            raise ValueError(f"Multiple folders found with images in {path}")
        else:
            return target_folders[0]
    except Exception as e:
        print(f"Error: {e}")
        return None

def find_file_by_ext(
    path: Union[str, Path], ext=[".json", ".csv"]
) -> Optional[Path]:
    try:
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Invalid path: {path}")
        if not path.is_dir() and not path.suffix in ext:
            raise ValueError(f"File does not have a valid extension: {path}")
        target_files = list(filter(lambda f: f.suffix in ext, path.rglob("*")))
        if len(target_files) == 0:
            raise ValueError(f"No files found with extensions {ext} in {path}")
        elif len(target_files) > 1:
            raise ValueError(f"Multiple files found with extensions {ext} in {path}")
        else:
            return target_files[0]
    except Exception as e:
        print(f"Error: {e}")
        return None