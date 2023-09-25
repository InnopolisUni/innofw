export HYDRA_FULL_ERROR=1
export experiments="classification/AK_130923_fbFMFDe1_lung_description_decision.yaml"
export data_source=$1
if [ -z "$data_source" ]
then
  python train.py "experiments=$experiments"
else
  python train.py "experiments=$experiments" \
                  "++datasets.test.source=$data_source"
fi
