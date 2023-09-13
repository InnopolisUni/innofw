export HYDRA_FULL_ERROR=1
export experiments="classification/AK_130923_fbFMFDe1_lung_description_decision.yaml"
export model_weights='/home/ainur/Desktop/innoolis/text/pipe.pkl'
#export model_weights="/home/ainur/git/innofw/logs/train/lung_description_decision/classification/AK_130923_fbFMFDe1_lung_description_decision.yaml/20230913-140214/checkpoints/model.pickle"
python infer.py "experiments=$experiments" \
                "++ckpt_path='$model_weights'" \


