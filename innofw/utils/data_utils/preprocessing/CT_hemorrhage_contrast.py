from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import cv2
import PIL
import logging
import sys
import pydicom

def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    return int(x)

def get_id(img_dicom):
    return str(img_dicom.SOPInstanceUID)

def get_metadata_from_dicom(img_dicom):
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
    import numpy as np
    # img = np.clip(img, min_val, max_val)
    img = np.clip(img, img_min, img_max)
    return img

def resize(img, new_w, new_h):
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img.resize((new_w, new_h), resample=PIL.Image.BICUBIC)

def save_img(img_pil, subfolder, name):
    img_pil.save(subfolder+name+'.png')

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    if mi == ma:
        return img/mi
    return (img - mi) / (ma - mi)

def prepare_image(img_dicom):
    img_id = get_id(img_dicom)
    metadata = get_metadata_from_dicom(img_dicom)
    img = window_image(img_dicom.pixel_array, **metadata)
    img = normalize_minmax(img) * 255

    # from pydicom.pixel_data_handlers.util import apply_modality_lut
    # ds2 = dicom_instance
    # arr2 = ds2.pixel_array  # Raw unitless pixel data
    # hu = apply_modality_lut(arr2, ds2)  # Pixel data has been converted to HU (for CT)
    # clipped_hu2 = (hu - np.min(hu)) / (np.max(hu) - np.min(hu))

    img = PIL.Image.fromarray(img.astype(np.uint8)).convert("L")
    return img_id, img

if __name__ == "__main__":
    try:
        import pydicom
        dicom_path2 = "/home/ainur/git/innofw/data/stroke/infer/images/1.dcm"
        dicom_path2 = "/home/ainur/data/rtk/images/1.2.392.200036.9116.2.6.1.48.1214245753.1506921568.925536/001.dcm"
        dicom_instance = pydicom.dcmread(dicom_path2)

        id, windowed = prepare_image(dicom_instance)

        # id, windowed = prepare_image(sys.argv[1])
        windowed =  np.array(windowed)
        new_path = dicom_path2[:-4] + id + ".png"
        from IPython import embed; embed()
        cv2.imwrite(new_path, windowed)
        # cv2.imwrite(sys.argv[2]+"/"+id.split("/")[-1], windowed)
        # cv2.imwrite(sys.argv[2]+"/"+id.split("/")[-1], windowed)
    except Exception as err:
        logging.error(err)
