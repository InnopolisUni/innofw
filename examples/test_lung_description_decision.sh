export HYDRA_FULL_ERROR=1
export experiments="classification/AK_130923_fbFMFDe1_lung_description_decision.yaml"
export model_weights="/home/ainur/Desktop/innopolis/text/pipe.pkl"
export data_source=$1
if [ -z "$data_source" ]
then
  python test.py "experiments=$experiments" \
                 "++ckpt_path=$model_weights"
else
  python test.py "experiments=$experiments" \
                 "++ckpt_path=$model_weights" \
                 "++datasets.test.source=$data_source"
fi