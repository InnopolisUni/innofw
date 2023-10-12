from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
import numpy as np
import os
import cv2
import joblib
import PIL
from glob import glob
import pydicom
import numpy as np
import pandas as pd
import os
import cv2
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm
import re
import logging as l
from glob import glob
import argparse

def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    return int(x)

def get_id(img_dicom):
    return str(img_dicom.SOPInstanceUID)

def get_metadata_from_dicom(img_dicom):
    metadata = {
        "window_center": 50,#img_dicom.WindowCenter,
        "window_width": 200,#img_dicom.WindowWidth,
        "intercept": -1024,#img_dicom.RescaleIntercept,
        "slope": 1.0,#img_dicom.RescaleSlope,
    }
    #print(metadata)
    return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

def window_image(img, window_center, window_width, intercept, slope):
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img 

def resize(img, new_w, new_h):
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img.resize((new_w, new_h), resample=PIL.Image.BICUBIC)

def save_img(img_pil, subfolder, name):
    img_pil.save(subfolder+name+'.png')

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)

def prepare_image(img_dicom):
    #img_dicom = pydicom.read_file(img_path)
    img_id = get_id(img_dicom)
    metadata = get_metadata_from_dicom(img_dicom)
    img = window_image(img_dicom.pixel_array, **metadata)
    img = normalize_minmax(img) * 255
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img_id, img

if __name__ == "__main__":
    try:
         id, windowed = prepare_image(sys.argv[1])
         windowed =  np.array(windowed)
         cv2.imwrite(sys.argv[2]+"/"+id, windowed)
    except:
         pass
