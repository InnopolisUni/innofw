# Uploading data and model checkpoints to s3
There are two ways for uploading: 
1. using python script
2. from cli (command line interface)

Both optinions will be considered.

## Uploading data 

Your data should be presented in the following way:

``` bash
data/
├── train/
│   └── train/
├── test/
│   └── test/
└── infer/
    └── infer/
```

### Python
For uploading your data to s3 you need to run the following code snippet.

```
from innofw.data_mart import upload_dataset


upload_dataset(dataset_config_path="config/datasets/classification)
```