export HYDRA_FULL_ERROR=1
export experiments="classification/AK_130923_fbFMFDe1_lung_description_decision.yaml"
python train.py "experiments=$experiments"
