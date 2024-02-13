from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from pydicom.pixel_data_handlers.util import apply_voi_lut
import PIL
import cv2
import numpy as np
import pydicom

from innofw.utils.data_utils.preprocessing.dicom_handler import dicom_to_raster

DEFAULT_PATH = "/home/ainur/git/innofw/data/stroke/infer/images/1.dcm"
DEFAULT_PATH = "/home/ainur/data/rtk/images/1.2.392.200036.9116.2.6.1.48.1214245753.1506921568.925536/001.dcm"


def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    return int(x)


def get_id(img_dicom):
    return str(img_dicom.SOPInstanceUID)


def get_metadata_from_dicom(img_dicom):
    # todo wtf argument?
    metadata = {
        "window_center": 50,
        "window_width": 200,
        "intercept": -1024,
        "slope": 1.0,
    }
    return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}


def window_image(img, window_center, window_width, intercept, slope):
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img


def resize(img, new_w, new_h):
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img.resize((new_w, new_h), resample=PIL.Image.BICUBIC)


def save_img(img_pil, subfolder, name):
    img_pil.save(subfolder + name + ".png")


def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    if mi == ma:
        return img / mi
    return (img - mi) / (ma - mi)


def prepare_image(img_dicom):
    img_id = get_id(img_dicom)
    metadata = get_metadata_from_dicom(img_dicom)
    img = window_image(img_dicom.pixel_array, **metadata)
    img = normalize_minmax(img) * 255
    img = PIL.Image.fromarray(img.astype(np.uint8)).convert("L")
    return img_id, img


def processing(input_path, output):
    dicom_instance = pydicom.dcmread(input_path)
    id_dcm, windowed = prepare_image(dicom_instance)
    windowed = np.array(windowed)
    windowed = dicom_to_raster(dicom_instance)
    if output is None:
        output = input_path[:-4] + id_dcm + ".png"
    cv2.imwrite(output, windowed)


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
