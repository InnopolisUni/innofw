# variables
experiment_name=KA_190722_0a70ef39_xgbregressor_cc
ckpt_path=https://api.blackhole.ai.innopolis.university/pretrained/credit_cards_xgb_best.pkl
# command
python infer.py experiments=$experiment_name +weights_path=$ckpt_path +ckpt_path=$ckpt_path
