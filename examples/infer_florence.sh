data_path=$1

if [ -z "$data_path" ]; then
  python infer.py experiments=detection/florence
else
  python infer.py experiments=detection/florence datasets.infer.source="$data_path"
fi
