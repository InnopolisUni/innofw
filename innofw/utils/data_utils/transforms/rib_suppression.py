from pathlib import Path

import cv2
import numpy as np
import torch
from pydantic import validate_arguments


class RibSuppression:
    """
    Class used for suppressing bone shadows in the Chest X-ray images.
    Attributes
    ----------
    equalize_out: bool
        a flag signalizing that the Histograms Equalization should be performed
    model_suppression: model
        rib suppression model

    Methods
    -------
    """

    @validate_arguments
    def __init__(self, equalize_out: bool = False):
        """
        Set a Histograms Equalization flag.

        :param equalize_out: a flag signalizing that the Histograms Equalization should be performed.
        """
        self.equalize_out = equalize_out
        self.model_suppression = load_bone_model_pytorch()

    def __call__(
        self,
        image,
        bbox=None,
        mask=None,
        keypoints=None,
        force_apply=False,
        *args,
        **kwargs,
    ):
        """
        Perform the ribs suppression in chest X-ray scans.

        :param image: an image that will be processed.
        :param bbox: a bounding box in pascal voc format, e.g. [x_min, y_min, x_max, y_max].
        :param mask: a mask for an input image.
        :param keypoints: a list of points in 'xy' format, e.g. [(x, y), ...].
        :return: a suppressed image.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Expected type 'numpy.ndarray', got "
                + type(image).__name__
                + "."
            )

        if len(np.shape(image)) > 2:
            # convert image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # get an image with suppressed bones
        suppressed = get_suppressed_image(
            image, self.model_suppression, equalize_out=self.equalize_out
        )
        return {
            "image": suppressed,
            "bbox": bbox,
            "mask": mask,
            "keypoints": keypoints,
        }


def load_bone_model_pytorch():
    """
    Load and return the trained model.
    """
    model = torch.jit.load(
        Path(__file__).parent.absolute() / "models/bone_suppression.pt",
        map_location=torch.device("cpu"),
    )
    model.eval()
    return model


def get_suppressed_image(image, model, equalize_out=False):
    """
    Perform bone suppression in an input image.

    :param image: an image, in which ribs should be suppressed.
    :param model: a pre-trained model for bone suppression.
    :param equalize_out: a flag signalizing that the Histograms Equalization should be performed.
    :return: an image, in which bone should be suppressed.
    """
    # create an array to store modified image
    img = np.copy(image)
    img_shape = img.shape
    if img.shape[0] % 8 != 0 or img.shape[1] % 8 != 0:
        new_shape = ((img.shape[0] // 8 + 1) * 8, (img.shape[1] // 8 + 1) * 8)
        img_temp = np.zeros(new_shape, dtype=np.uint8)
        img_temp[: img_shape[0], : img_shape[1]] = img
        img = img_temp
    img_torch = (
        torch.from_numpy(img.copy())
        .unsqueeze(0)
        .unsqueeze(1)
        .type(torch.float32)
    )
    # get the result of suppression
    with torch.no_grad() as tn:
        pred = model(img_torch)
    res = pred[0, 0, : img_shape[0], : img_shape[1]].numpy()
    res = np.clip(np.round(res), 0, 255).astype(np.uint8)
    # perform the Histogram Equalization over the image
    if equalize_out:
        res = cv2.equalizeHist(res.astype(np.uint8))
    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process input/output image paths"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="path to lung image in png format",
    )
    parser.add_argument(
        "-o", "--output_path", required=True, help="path to save image"
    )
    argv = parser.parse_args()
    img = cv2.imread(argv.input_path)
    suppress = RibSuppression()
    suppressed = suppress(img)["image"]
    cv2.imwrite(argv.output_path, suppressed)
