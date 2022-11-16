mkdir $(pwd)/data/stroke_png
python innofw/utils/data_utils/transforms/db_scan.py \
-i $(pwd)/tests/data/images/images_for_sh/medicine/1.jpg -o $(pwd)/data/stroke_png/contrasted.png

