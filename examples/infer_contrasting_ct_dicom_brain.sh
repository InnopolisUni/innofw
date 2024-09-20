data_path=$1
out_path=$2

if [ -z "$data_path" ]; then
	data_path="../innofw/data/rtk/infer/"
	echo "Using default data path $data_path"
fi

python innofw/utils/data_utils/preprocessing/CT_hemorrhage_contrast.py  --input "$data_path" --output "$out_path"