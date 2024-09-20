from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import os

from tqdm import tqdm
import cv2
import numpy as np

from innofw.core.datasets.coco import DicomCocoDataset_rtk
from innofw.utils.data_utils.rtk.CT_hemorrhage_metrics import transform


def hemorrhage_contrast(input_path: str, output_folder: str = None):
    if output_folder is None or output_folder == "":
        output_folder = default_output_path()
    if type(output_folder) != str:
        try:
            output_folder = str(output_folder)
        except TypeError:
            raise ValueError(f"Wrong path to save: {output_folder}")

    dataset = DicomCocoDataset_rtk(data_dir=input_path, transform=transform)
    if len(dataset) == 0:
        raise Warning(f"empty dataset with the directory {input_path}")
    else:
        for x in (pbar := tqdm(dataset)):
            path = x["path"]
            mask = x["mask"]
            contrasted_image = x["image"]
            raw_image = x["raw_image"]

            basename = Path(path).stem
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(output_folder, basename + "_raw.npy")
            try:
                np.save(output_path, raw_image)
            except:
                pbar.set_description(f"wrong path {output_path}")

            output_path = os.path.join(output_folder, basename + "_mask.png")
            if not cv2.imwrite(output_path, mask):
                pbar.set_description(f"wrong path {output_path}")

            output_path = os.path.join(output_folder, basename + "_image.png")
            if not cv2.imwrite(output_path, contrasted_image):
                pbar.set_description(f"wrong path {output_path}")


def callback(arguments):
    """Callback function for arguments"""
    try:
        hemorrhage_contrast(arguments.input, arguments.output)
    except KeyboardInterrupt:
        print("You exited")


def default_output_path() -> str:
    from innofw.utils.getters import get_log_dir
    from uuid import uuid4

    log_root = os.path.join(os.getcwd(), "logs")
    project = "contrast"
    stage = "infer"
    experiment_name = str(uuid4()).split("-")[0]
    return str(get_log_dir(project, stage, experiment_name, log_root))


def setup_parser(parser):
    """The function to setup parser arguments"""
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="path to dataset to load, default path is %(default)s",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="path to dataset to save",
    )


def main():
    """Main module function"""
    parser = ArgumentParser(
        prog="hemorrhage_contrast",
        description="A tool to contrast",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    callback(arguments)


if __name__ == "__main__":
    main()
