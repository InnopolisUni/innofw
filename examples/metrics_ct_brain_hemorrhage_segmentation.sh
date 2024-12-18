data_path=$1
output=$2
if [ -z "$data_path" ]; then
	data_path="../innofw/data/rtk/infer/"
	echo "Using default data path $data_path"
fi

if [ -z "$output" ]; then
	output="../innofw/logs/infer/segmentation/semantic-segmentation/SK_180822_qmciwj41_unet_brain_rtk/"
	output+="$(ls $output -tr| tail -n 1)"
	echo "Using default output path $output"
fi

python innofw/utils/data_utils/rtk/CT_hemorrhage_metrics.py -i "$data_path" -o "$output" -t "segmentation"
