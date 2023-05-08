access_key=one
secret_key=two

#python innofw/zoo/uploader.py --ckpt_path pretrained/best.pkl\
#                                     --config_save_path config/models/result.yaml\
#                                     --remote_save_path https://api.blackhole.ai.innopolis.university/pretrained/testing/lin_reg_house_prices.pickle\
#                                     --access_key $access_key --secret_key $secret_key\
#                                     --target sklearn.linear_models.LinearRegression --data some/path/to/data\
#                                     --description "linear regression model trained on house prices dataset"\
#                                     --metrics '{"mse": 0.04}'\
#                                     --name something