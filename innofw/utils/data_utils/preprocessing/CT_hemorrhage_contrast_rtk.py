from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from urllib.parse import urlparse
import os

from tqdm import tqdm
import cv2
import numpy as np


from innofw.core.datamodules.lightning_datamodules.coco_rtk import (
    DicomCocoDataModuleRTK,
)
from innofw.utils.data_utils.rtk.CT_hemorrhage_metrics import transform


def hemorrhage_contrast(input_path: str, output_folder: str = None):
    if output_folder is None or output_folder == "":
        output_folder = default_output_path()
    if type(output_folder) != str:
        try:
            output_folder = str(output_folder)
        except TypeError:
            raise ValueError(f"Wrong path to save: {output_folder}")
    if urlparse(input_path):
        default_path = "./data/rtk/infer/"
        path = {"source": input_path, "target": default_path}
    else:
        path = {"source": input_path, "target": input_path}
    dm = DicomCocoDataModuleRTK(infer=path, transform=transform)
    dm.setup_infer()
    dataloader = dm.predict_dataloader()
    dataset = dataloader.dataset
    dataset.transform = transform

    if len(dataset) == 0:
        raise Warning(f"empty dataset with the directory {input_path}")
    else:
        for x in (pbar := tqdm(dataset)):
            path = x["path"]
            mask = x.get("mask", None)
            contrasted_image = x["image"]
            raw_image = x.get("raw_image", None)

            basename = Path(path).stem
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(output_folder, basename + "_raw.npy")
            np.save(output_path, raw_image)

            output_path = os.path.join(output_folder, basename + "_mask.png")
            cv2.imwrite(output_path, mask)

            output_path = os.path.join(output_folder, basename + "_image.png")
            cv2.imwrite(output_path, contrasted_image)


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
