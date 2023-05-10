# Uploading data and model checkpoints to s3
There are two ways to upload them: 
1. using python script
2. from cli (command line interface)

Both optinions will be considered.

For both uploading data and model using either cli or python script you can omit access_key and secret_key arguments, but they will be asked. So you need to specify them anyway.

## Uploading data 
To upload data you need to have the dataset config file that must be stored in <b>config/datasets</b> folder. Here is the example:
``` yaml
# classification/classification_mnist.yaml
task:
  - image-classification

name: mnist
description: "description"

markup_info: markup info
date_time: 01.08.2022

_target_: innofw.core.datamodules.lightning_datamodules.image_folder_dm.ImageLightningDataModule

train:
  target: ./data/mnist/train
test:
  target: ./data/mnist/test
infer:
  target: ./data/mnist/infer
```

The necessary fields are train and test target. These paths must contain train and test folders inside correspondingly. The data folder for example above looks like this:

``` bash
data/
└── mnist/
    ├── train/
    │   └── train/
    ├── test/
    │   └── test/
    └── infer/
        └── infer/

```

### Python
To upload your data to s3 you need to run the following code snippet with appropriate values.

```
from innofw.data_mart import upload_dataset


upload_dataset(dataset_config_path="classification/config.yaml",
                remote_save_path="https://api.blackhole.ai.innopolis.university/public-datasets/test_dataset/",
                access_key = "access key",
                secret_key = "secret key")
```

### CLI
To upload your data to s3 you can also run this command in cli.

``` bash
python innofw/data_mart/uploader.py --dataset_config_path classification/config.yaml
                                    --remote_save_path  https://api.blackhole.ai.innopolis.university/public-datasets/test_dataset/ 
                                    --access_key access_key --secret_key secret_key\
```
## Uploading model checkpoint to s3
To upload your checkpoint you need to specify config file of your experiment that must be stored in <b>config/experiments</b> folder, path to checkpoint and metrics you want to save.
### Python
You can run this python script to upload checkpoint with appropriate values.
``` python
from innofw.zoo import upload_model


upload_model(experiment_config_path="classification/config.yaml",
            ckpt_path = "pretrained/best.pkl",
            remote_save_path = "https://api.blackhole.ai.innopolis.university/pretrained/model.pickle",
            metrics = {"some metric": 0.04},
            access_key = "access key",
            secret_key = "secret key")
```
### CLI
To upload your checkpoint you can also run this command.
``` bash
python innofw/zoo/uploader.py --ckpt_path pretrained/best.pkl
                              --experiment_config_path classification/config.yaml
                              --remote_save_path https://api.blackhole.ai.innopolis.university/pretrained/testing/lin_reg_house_prices.pickle
                              --access_key access_key --secret_key secret_key
```