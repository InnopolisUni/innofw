data_path=$1
out_path=$2
python innofw/utils/data_utils/rtk/CT_hemorrhage_metrics.py -i "$data_path" -o "$out_path" -t "detection"