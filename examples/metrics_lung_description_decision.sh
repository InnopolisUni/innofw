data_path=$1
if [ -z "$data_path" ]; then
	data_path="./data/lung_description/infer/labels.csv"
	echo "Using default data path $data_path"
fi

if [ -z "$output" ]; then
	output="./logs/infer/lung_description_decision/classification/AK_130923_fbFMFDe1_lung_description_decision.yaml/"
	output+="$(ls $output -tr| tail -n 1)/"
	output+="$(ls $output -tr| tail -n 1)"
	echo "Using default output path $output"
fi

python innofw/utils/data_utils/rtk/lungs_description_metrics.py -i "$data_path" -o "$output"
