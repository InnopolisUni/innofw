# create train, test and infer folders
# with images and labels folders inside
import shutil
from pathlib import Path
import logging

from fire import Fire
from pydantic import validate_arguments, DirectoryPath, FilePath


@validate_arguments
def prepare_dataset(
    src_img_path: DirectoryPath,
    src_label_path: DirectoryPath,
    train_names_file: FilePath,
    test_names_file: FilePath,
    dst_path: Path,
):
    with open(train_names_file, "r") as f:
        train_names = [i for i in f.read().split("\n") if i != ""]

    with open(test_names_file, "r") as f:
        test_names = [i for i in f.read().split("\n") if i != ""]

    sets = ["train", "test", "infer"]
    names = [train_names, test_names, test_names]

    for s, n in zip(sets, names):
        set_dst_path = dst_path / s
        set_dst_path.mkdir(exist_ok=True, parents=True)
        set_img_dst_path = set_dst_path / "images"
        set_label_dst_path = set_dst_path / "labels"

        set_img_dst_path.mkdir(exist_ok=True, parents=True)
        set_label_dst_path.mkdir(exist_ok=True, parents=True)

        for filename in n:
            try:
                img_name = f"{Path(filename).name}.jpeg"
                shutil.copy(
                    src_img_path / img_name,
                    set_img_dst_path / img_name,
                )
                label_name = f"{Path(filename).name}.txt"
                shutil.copy(
                    src_label_path / label_name,
                    set_label_dst_path / label_name,
                )
            except:
                logging.warning(f"Could not copy file: {filename}")


if __name__ == "__main__":
    Fire(prepare_dataset)
