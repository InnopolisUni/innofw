import sys
import pydicom
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
import PIL
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt

from innofw.utils.data_utils.preprocessing.dicom_handler import dicom_to_raster

import logging
logger = logging.getLogger()
logger.handlers = []


def output_path(default_input_path: str):
    print("INPUT:", default_input_path)
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
        ValueError("no such path")
    default_output_path = None
    return default_output_path

OUTPUT_PATH = None



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


def apply_window_level(image, window_width=350, window_level=40):
    min_intensity = window_level - (window_width / 2)
    max_intensity = window_level + (window_width / 2)

    # Clip the image based on the window range
    image = np.clip(image, min_intensity, max_intensity)

    # Normalize the image to range [0, 255]
    image = ((image - min_intensity) / (max_intensity - min_intensity) * 255).astype(
        np.uint8
    )
    return image


def prepare_image(img_dicom):
    img_id = get_id(img_dicom)
    metadata = get_metadata_from_dicom(img_dicom)
    img = window_image(img_dicom.pixel_array, **metadata)
    img = normalize_minmax(img) * 255
    img = PIL.Image.fromarray(img.astype(np.uint8)).convert("L")
    return img_id, img


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
    colored_mask = cv2.resize(colored_mask.astype(np.uint8), shape_to, interpolation=cv2.INTER_NEAREST)
    overlayed_image = image.copy()
    overlayed_image[colored_mask.astype(bool)] = color  # Применение цвета только на маске

    return overlayed_image


from innofw.core.datasets.coco import DicomCocoDataset_sm
from tqdm import tqdm

import albumentations as A

transform = A.Compose([
    # A.Resize(256, 256),  # Изменение размера для изображения и маски
])

def processing(input_path, output_folder=OUTPUT_PATH):



    dataset = DicomCocoDataset_sm(data_dir=input_path, transform=transform)
    for x in (pbar := tqdm(dataset)):
        path = x["path"]
        mask = x["mask"]
        image = x["image"]
        raw_image = x["raw_image"]
        output_folder = "/home/ainur/data/rtk/pictures_only/"

        os.makedirs(output_folder, exist_ok=True)

        basename = Path(path).stem
        # contrasted_img = apply_window_level(image)
        contrasted_img = image
        cv2.imwrite(os.path.join(output_folder, basename + ".png"), image)
        contrasted_image = overlay_mask_on_image(contrasted_img, mask)
        # output_path = os.path.join(output_folder, basename + ".png")
        f, ax = plt.subplots(1, 2)

        # ax[0].imshow(raw_image, cmap="Greys_r")
        # ax[1].imshow(contrasted_image)
        plt.show()

        # if cv2.imwrite(output_path, contrasted_image):
        #     pbar.set_description(f"saved as {output_path}")
        # else:
        #     pbar.set_description(f"wrong path {output_path}")


def other_methods_to_do_this(path, use_innofw=True):
    """contrasted to apply_window_level

    Args:
        path:
        use_innofw:

    Returns:

    """
    dicom_instance = pydicom.dcmread(path)
    if not use_innofw:
        id_dcm, windowed = prepare_image(dicom_instance)
        windowed = np.array(windowed)
    else:
        windowed = dicom_to_raster(dicom_instance)
    return windowed


def callback(arguments):
    """Callback function for arguments"""
    try:
         id, windowed = prepare_image(sys.argv[1])
         windowed =  np.array(windowed)
         cv2.imwrite(sys.argv[2]+"/"+id.split("/")[-1], windowed)
    except Exception as err:
        l.error(err)
