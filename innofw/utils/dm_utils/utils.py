import os
import sys
from pathlib import Path
from typing import Union, Optional
from pydantic import validate_arguments, DirectoryPath
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

@validate_arguments
def find_folder_with_images(
    path: DirectoryPath,
    img_exts=[".jpg", ".png"]
) -> Optional[Path]:
    img_exts = img_exts if type(img_exts) == list else [img_exts]
    try:
        target_folders = []
        for ext in img_exts:
            target_folders += list(filter(lambda f: f.is_dir() and any(f.glob(f'*{ext}')), path.iterdir()))
        target_folders = list(set(target_folders)) # remove duplicates
        if len(target_folders) == 0:
            raise ValueError(f"No folders found with images in {path}")
        elif len(target_folders) > 1:
            raise ValueError(f"Multiple folders found with images in {path}")
        else:
            return target_folders[0]
    except Exception as e:
        print(f"Error: {e}")
        return None
    
@validate_arguments
def find_file_by_ext(
    path: DirectoryPath, ext=[".json", ".csv"]
) -> Optional[Path]:
    exts = ext if isinstance(ext, list) else [ext]
    if path.is_file():
        if path.suffix in exts:
            return path
        else:
            return None
    if not path.is_dir():
        print(f"Error: {path} is not a valid directory or file path")
        return None
    try:
        target_files = list(filter(lambda f: f.suffix in exts, path.rglob("*")))
        return check_files(path, ext, target_files)
    except ValueError as e:
        print(f"Error: {e}")
        return None

def check_files(path: Path, ext, target_files):
    if len(target_files) == 0:
        return None
    elif len(target_files) > 1:
        print(f"Error: Multiple files found in {path} with extensions {ext}:")
        for file_path in target_files:
            print(f"- {file_path}")
        return None
    else:
        return target_files[0]

