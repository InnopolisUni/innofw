data_path=$1
out_path=$2

if [ -z "$data_path" ]; then
	data_path="https://api.blackhole.ai.innopolis.university/public-datasets/rtk/infer.zip"
	echo "Using default data path $data_path"
fi

python innofw/utils/data_utils/preprocessing/CT_hemorrhage_contrast_rtk.py  --input "$data_path" --output "$out_path"