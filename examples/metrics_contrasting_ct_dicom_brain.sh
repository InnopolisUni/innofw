output=$1
if [ -z "$output" ]; then
	output="../innofw/logs/infer/contrast"
	output+="/$(ls $output -tr | tail -n 1)"
	output+="/$(ls $output -tr | tail -n 1)"
fi
python innofw/utils/data_utils/preprocessing/CT_hemorrhage_contrast_metrics.py -o "$output"
