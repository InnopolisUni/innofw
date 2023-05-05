access_key=one
secret_key=two

#python innofw/zoo/uploader.py --ckpt_path pretrained/best.pkl\
#                                     --model_config_path config/models/classification/satellite_imagery
#                                     --data_folder ./data/UCMerced/
#                                     --remote_save_path https://api.blackhole.ai.innopolis.university/pretrained/testing/lin_reg_house_prices.pickle\
#                                     --access_key $access_key --secret_key $secret_key\
#                                     --metrics '{"mse": 0.04}'\