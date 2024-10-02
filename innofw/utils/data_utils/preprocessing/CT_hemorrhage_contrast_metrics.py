from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import cv2
import numpy as np


def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Наложение маски на изображение с цветовой кодировкой для каждого класса без изменения оригинальных цветов изображения.

    Args:
        image (np.array): Исходное изображение (HxWxC).
        mask (np.array): Маска в формате (H, W, D), где D - количество классов.
        alpha (float): Прозрачность маски.

    Returns:
        np.array: Изображение с наложенной маской.
    """
    color = np.array([255, 0, 0])

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = np.stack([image] * 3, axis=-1)

    colored_mask = np.any(mask > 0, axis=-1)
    shape_to = image.shape[:2]
    colored_mask = cv2.resize(
        colored_mask.astype(np.uint8), shape_to, interpolation=cv2.INTER_NEAREST
    )
    overlayed_image = image.copy()
    overlayed_image[colored_mask.astype(bool)] = (
        overlayed_image[colored_mask.astype(bool)] * (1 - alpha) + alpha * color
    )
    return overlayed_image


def calculate_metrics(raw, contrasted):

    # PSNR
    psnr = peak_signal_noise_ratio(raw, contrasted)

    # SSIM
    ssim = structural_similarity(raw, contrasted)

    return {"Peak Signal-to-Noise Ratio": psnr, "Structural Similarity Index": ssim}


def hemorrhage_contrast_metrics(input_path: str):
    try:
        input_path = str(input_path)
    except TypeError:
        raise Exception(f"wrong path {input_path}")

    assert os.path.exists(input_path), f"wrong path {input_path}"
    files = os.listdir(input_path)
    uniqes = [x.rsplit("_", 1)[0] for x in files]

    for f in uniqes:
        mask = cv2.imread(os.path.join(input_path, f + "_mask.png"), 0)
        raw_image = np.load(os.path.join(input_path, f + "_raw.npy"))
        image = cv2.imread(os.path.join(input_path, f + "_image.png"), 0)
        mask = np.expand_dims(mask, 2)
        contrasted_image_with_mask = overlay_mask_on_image(image, mask)

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(raw_image, cmap="Greys_r")
        ax[1].imshow(contrasted_image_with_mask)

        metrics = calculate_metrics(raw_image, image)
        plt.suptitle("\n".join([f"{k}:{np.round(v, 2)}" for k, v in metrics.items()]))
        plt.show()


def callback(arguments):
    """Callback function for arguments"""
    try:
        hemorrhage_contrast_metrics(arguments.input)
    except KeyboardInterrupt:
        print("You exited")


def setup_parser(parser):
    """The function to setup parser arguments"""
    parser.add_argument(
        "-i",
        "--input",
        "-o",
        "--output",
        required=True,
        help="path to dataset to load, default path is %(default)s",
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
