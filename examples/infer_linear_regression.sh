# variables
experiment_name=regression/KA_130722_9f7134db_linear_regression
ckpt_path=https://api.blackhole.ai.innopolis.university/pretrained/house_prices_lin_reg.pickle
# command
python infer.py experiments=$experiment_name +ckpt_path=$ckpt_path
