mkdir $(pwd)/data/stroke_png
python3 innofw/utils/data_utils/preprocessing/dicom_handler.py $(pwd)/tests/data/images/images_for_sh/medicine/1.dcm $(pwd)/data/stroke_png/1.jpg