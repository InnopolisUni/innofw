import datetime
import os.path

import fire
import numpy as np
from PIL import Image
from pydantic.types import FilePath
from pydicom import Dataset
from pydicom import dcmread
from pydicom.pixel_data_handlers import apply_modality_lut
from pydicom.pixel_data_handlers import apply_voi_lut
from pydicom.uid import generate_uid
import pydicom
from innofw.utils import get_project_root


def img_to_dicom(
    img: np.ndarray,
    origin_dicom: FilePath = None,
    path_to_save: FilePath = None,
) -> Dataset:
    """
    Converts image array to dicom dataset

    Args:
         img: image array with 1 or 3 channels
         path_to_save: path to save Dicom. If None dicom would not be saved.
         origin_dicom: path to original dicom file to copy metadata.
    Returns:
        dicom Dataset
    """
    dicom = raster_to_dicom(img, origin_dicom)
    if path_to_save:
        dicom.save_as(path_to_save)
    return dicom


def dicom_to_img(dicom: FilePath, path_to_save: FilePath = None) -> np.ndarray:
    """
    Converts dicom file to img array

    Args:
         dicom: path to dicom
         path_to_save: path to save image. Can be saved in any extension. If None image would not be saved.
    Returns:
        numpy array with 3 channels image
    """
    try:
        dataset = dcmread(dicom)
    except IsADirectoryError:
        return
    img = dicom_to_raster(dataset)
    if path_to_save:
        im = Image.fromarray(img)
        im.save(path_to_save)
    return img


def crate_base_dataset() -> Dataset:
    dataset = dcmread(
        os.path.join(
            get_project_root(),
            "innofw/utils/data_utils/preprocessing/src/tmp.dcm",
        )
    )
    dataset.SOPInstanceUID = generate_uid()
    dataset.PatientName = "Anon"
    dataset.PatientID = "123456"
    dataset.SeriesInstanceUID = generate_uid()
    dataset.StudyInstanceUID = generate_uid()
    dataset.is_implicit_VR = True
    dataset.is_little_endian = True
    dataset.SOPClassUID = generate_uid()
    dataset.SamplesPerPixel = 1
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    dataset.file_meta = file_meta
    dataset.add_new((0x0008, 0x0000), "UL", 128)
    dataset.add_new((0x0010, 0x0000), "UL", 18)
    dataset.add_new((0x0020, 0x0000), "UL", 72)
    dataset.add_new((0x0028, 0x0000), "UL", 1368)
    dataset.SpecificCharacterSet = "ISO_IR 192"
    dt = datetime.datetime.now()
    dataset.ContentDate = dt.strftime("%Y%m%d")
    timeStr = dt.strftime("%H%M%S.%f")  # long format with micro seconds

    dataset.ContentTime = timeStr
    dataset.WindowCenter = None
    dataset.WindowWidth = None

    return dataset


def dicom_to_raster(dataset: Dataset) -> np.ndarray:
    try:
        img = apply_modality_lut(
            apply_voi_lut(dataset.pixel_array, dataset), dataset
        )
    except IndexError:
        img = apply_modality_lut(dataset.pixel_array, dataset)
    except Exception as err:
        raise ValueError(f"Could not convert to raster: {err}")
    image = img.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return (image * 255).astype("uint8")


def raster_to_dicom(img: np.ndarray, dicom: FilePath = None) -> Dataset:
    if not dicom:
        dataset = crate_base_dataset()
    else:
        dataset = dcmread(dicom)
        dataset.SOPInstanceUID = generate_uid()
    dataset = add_image(dataset, img)
    return dataset


def add_image(ds: Dataset, img: np.ndarray) -> Dataset:
    img = Image.fromarray(img)
    if img.mode == "L":
        # (8-bit pixels, black and white)
        np_frame = np.array(img.getdata(), dtype=np.uint8)
        ds.Rows = img.height
        ds.Columns = img.width
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.SamplesPerPixel = 1
        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = pydicom.encaps.encapsulate([np_frame.tobytes()])
    elif img.mode == "RGBA" or img.mode == "RGB":
        # RGBA (4x8-bit pixels, true colour with transparency mask)
        np_frame = np.array(img.getdata(), dtype=np.uint8)[:, :3]
        ds.Rows = img.height
        ds.Columns = img.width
        ds.PhotometricInterpretation = "RGB"
        ds.SamplesPerPixel = 3
        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = np_frame.tobytes()
    return ds


if __name__ == "__main__":
    fire.Fire(dicom_to_img)
