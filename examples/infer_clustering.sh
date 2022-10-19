# variables
experiment_name=KA_280722_9379c32c_kmeans_credit_cards
ckpt_path=https://api.blackhole.ai.innopolis.university/pretrained/credit_cards_kmeans.pickle
dataset_conf_name=clustering_credit_cards_infer
# command
python infer.py experiments=$experiment_name datasets=$dataset_conf_name +weights_path=$ckpt_path +ckpt_path=$ckpt_path
