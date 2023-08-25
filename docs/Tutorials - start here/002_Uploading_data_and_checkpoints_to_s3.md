# Uploading Data and Model Checkpoints to Amazon S3
There are two methods for uploading data and model checkpoints to Amazon S3: using the API and the command line interface (CLI). Both options will be covered in this tutorial.

When uploading data or a model using either the command line interface (CLI) or an API, you have the option to omit the access_key and secret_key arguments. The framework will search for these values in the ~/.aws/credentials and .env files. However, if these values are not found, you will be prompted in CLI.

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

### API
To upload data using the API use the provided Python code snippet, specifying the appropriate values.

``` python 
from innofw.data_mart import upload_dataset


upload_dataset(dataset_config_path="classification/config.yaml",
                remote_save_path="https://api.blackhole.ai.innopolis.university/public-datasets/test_dataset/",
                access_key = "access key",
                secret_key = "secret key")
```

### CLI
Alternatively, you can upload data to S3 using the CLI. Run the following command, replacing the placeholders with the appropriate values.

``` bash
python innofw/data_mart/uploader.py --dataset_config_path classification/config.yaml \
                                    --remote_save_path  https://api.blackhole.ai.innopolis.university/public-datasets/test_dataset/ \
                                    --access_key access_key --secret_key secret_key
```
## Uploading model checkpoint to s3
To upload your checkpoint, you need to specify the config file for your experiment, which should be stored in the config/experiments folder. Additionally, provide the path to the checkpoint and specify the metrics you want to save.
### API
You can call this Python function to upload a checkpoint with appropriate values.
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
You can also use the CLI to upload your model checkpoint. Run the following command, replacing the placeholders with the appropriate values.
``` bash
python innofw/zoo/uploader.py --ckpt_path pretrained/best.pkl\
                              --experiment_config_path classification/config.yaml\
                              --remote_save_path https://api.blackhole.ai.innopolis.university/pretrained/testing/lin_reg_house_prices.pickle\
                              --metrics '{"some metric": 0.04}'\
                              --access_key access_key --secret_key secret_key
```