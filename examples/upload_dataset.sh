export access_key=one
export secret_key=two
python innofw/data_mart/uploader.py \
    --dataset_config_path semantic-segmentation/arable-bin-seg-ndvi/fields/tile-512-90-days.yaml \
    --remote_save_path https://api.blackhole.ai.innopolis.university/public-datasets/testing/arable_ndvi/ \
    --access_key $access_key \
    --secret_key $secret_key