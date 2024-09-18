
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import os

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import cv2

from innofw.core.datasets.coco import DicomCocoDataset_sm
from innofw.utils.data_utils.preprocessing.CT_hemorrhage_contrast import (
    DEFAULT_PATH,
    OUTPUT_PATH,
)


def calculate_metrics(raw, contrasted):
    if len(raw.shape) ==2 or raw.shape[2] == 1:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)

    # PSNR
    psnr = peak_signal_noise_ratio(raw, contrasted)

    # SSIM
    ssim = structural_similarity(raw, contrasted, channel_axis=2)

    # # Cosine Similarity
    # # Преобразуем изображения в векторы
    # image1_vector = image1.flatten().reshape(1, -1)
    # image2_vector = image2.flatten().reshape(1, -1)
    #
    # # Вычисляем косинусное сходство
    # cos_sim = cosine_similarity(image1_vector, image2_vector)[0][0]

    return psnr, ssim

def output_path(default_input_path: str):
    default_input_path = str(default_input_path)  # in case of path-objects
    if os.path.exists(default_input_path):
        # assert os.path.isdir(default_input_path), "there should be a dir to a collection of DICOM"
        if default_input_path.endswith("/"):
            default_input_path = default_input_path[:-1]
        *parts, last_part = default_input_path.split(os.path.sep)
        new_path_parts = parts + [last_part + "_output"]
        default_output_path = os.path.join(*new_path_parts)
        os.makedirs(default_output_path, exist_ok=True)
    else:
        ValueError("no such ")
    return default_output_path


OUTPUT_PATH = output_path(DEFAULT_PATH)


def processing(input_path, output_folder=OUTPUT_PATH):

    dataset = DicomCocoDataset_sm(data_dir=input_path)
    for x in (pbar := tqdm(dataset)):
        path = x["path"]
        image = x["image"]
        image = image[:, :, 0]

        os.makedirs(output_folder, exist_ok=True)
        basename = Path(path).stem
        output_path = os.path.join(output_folder, basename + ".png")
        if os.path.exists(output_path):
            image2 = cv2.imread(output_path)
            psnr, ssim =  calculate_metrics(image, image2)
            print(f"Calculated metrics for images {path} and {output_path}:\n"
                  f"Peak Signal-to-Noise Ratio: {psnr}\n"
                  f"Structural Similarity Index: {ssim}\n"
                  )
        else:
            print(f"file {output_path} does not exist")


def callback(arguments):
    """Callback function for arguments"""
    return processing(arguments.input, arguments.output)


def setup_parser(parser):
    """The function to setup parser arguments"""
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_PATH,
        help="path to dataset to load, default path is %(default)s",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=OUTPUT_PATH,
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
