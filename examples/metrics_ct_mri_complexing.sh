data_path=$1
output=$2

if [ -z "$data_path" ]; then
	data_path="../innofw/data/complex/infer"
	echo "Using default data path $data_path"
fi

if [ -z "$output" ]; then
	output="../innofw/logs/infer/segmentation/semantic-segmentation/SK_100923_unet_brain_complex.yaml/"
	output+="$(ls $output -tr | tail -n 1)"
fi
python innofw/utils/data_utils/rtk/CT_complexing_metrics.py -i $data_path -o $output
