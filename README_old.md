## Requirements

- Linux|Windows operating system
- 30 GB+ of storage, 4 GB+ RAM
- Python 3.8+, <3.10
- Poetry 1.1+

## Section 1. Configuring Loss Function

Let's take a look at the following .yaml file

```yaml

task:
  - image-segmentation

implementations:
  torch: # framework name
    JaccardLoss: # name of the loss function(can any name)
      weight: 0.3  # weight of the loss function(can be any float)
      object: # can be `function`
        # path to the class/function(can be a local path or a installed library code)
        _target_: pytorch_toolbelt.losses.JaccardLoss
        mode: binary  # additional argument for the class
    BinaryFocalLoss:  # another loss, structure is similar to above described loss 
      weight: 0.7
      object:
        _target_: pytorch_toolbelt.losses.BinaryFocalLoss
```

`task` denotes the type of the problem these loss functions were designed to be used.

`implementations` contains information on how to instantiate the loss functions for different frameworks.

Inner level is for framework names. Here we can use `torch`, `sklearn`, `xgboost` etc.
Inside of the framework level we have the names of the objects. Names are later used during logging.
You are free to select any name.

Latter if we go inside the "name's level" we will have two fields: weight, object/function.
Weight is used to specify the weight of the loss function.

#### Object/Function:

**TL;DR**

- if code to be instantiated is a function then name this field `function`
- if code to be instantiated is an object then name this field `object`


---
Here we are choosing the type of the code we want to instantiate.

It can be an `object` of a class or a `function`.
As functions cannot be instantiated right away without arguments.
We need to instantiate function later in the code when we receive arguments.

Under the hood:

    object - gets instantiated
    function - gets wrapped into a lambda function

this allows us to have the same interface for both objects and functions later on.

Example:

In the following snippet we initialize the loss object `BinaryFocalLoss`

```python
from pytorch_toolbelt.losses import BinaryFocalLoss
import torch

criterion = BinaryFocalLoss()

pred = torch.tensor([0.0, 0.0, 0.0])
target = torch.tensor([1, 1, 1])

pred.unsqueeze_(0)
target.unsqueeze_(0)

loss1 = criterion(pred, target)
```

In the following snippet we initialize the function `binary_cross_entropy` and pass arguments right away.

```python
import torch
import torch.nn.functional as F

pred = torch.tensor([0.0, 0.0, 0.0])
target = torch.tensor([1, 1, 1])

pred.unsqueeze_(0)
target.unsqueeze_(0)

loss1 = F.binary_cross_entropy(pred, target)
```

## Section 2. How to add your dataset?

Now we will consider adding your custom dataset into the framework.

1. Split your data into two folders: train and test.
2. Make sure that you have the corresponding datamodule to process your data. All the available datamodules stored in
   `innofw/core/datamodules/`. Each datamodule has a `task` and `framework` attributes*. Pair of `task` and `framework` can
   be duplicated, in this case difference is
   in the data retrieval logic, select one that is more suitable for your problem.
    1. In case you have not found suitable datamodule then write your own. Refer
       to [section 2.2](#Section-2.2.-Writing-own-datamodule).
3. Create a configuration file in config/datasets/[dataset_name].yaml\
   Dataset config file should be structured as follows:
   ```yaml
   
      task:
        - [dedicated task]

      name: [name of the dataset]
      description: [specify dataset description]
   
      markup_info: [specify markup information]
      date_time: [specify date]
   
      _target_: innofw.core.datamodules.[submodule].[submodule].[class_name]
   
    # =============== Data Paths ================= #
    # use one of the following:

      # ====== 1. local data ====== #
      train:
        source: /path/to/file/or/folder
      test:
        source: /path/to/file/or/folder   
      # ====== 2. remote data ====== #
      train:
        source: https://api.blackhole.ai.innopolis.university/public-datasets/folder/train.zip
        target: folder/to/extract/train/
      test:
        source: https://api.blackhole.ai.innopolis.university/public-datasets/folder/test.zip
        target: folder/to/extract/test/
    # ================================== #
           
      # some datamodules require additional arguments
      # look for them in the documentation of each datamodule
      # arguments passed in the following way:
      arg1: value1  # here arg1 - name of the argument, value1 - value for the arg1
      arg2: value2
      # ... same for other datamodule arguments

4. To run prediction on new data you should create an inference datamodule configuration file. Configuration file is
   alike to file created in 3.
   ```yaml
   
      task:
        - [dedicated task]

      name: [name of the dataset]
      description: [specify dataset description]
   
      markup_info: [specify markup information]
      date_time: [specify date]
   
      _target_: innofw.core.datamodules.[submodule].[submodule].[class_name]
   
    # =============== Data Paths ================= #
    # use one of the following:

      # ====== 1. local data ====== #
      infer:
        source: /path/to/file/or/folder   
      # ====== 2. remote data ====== #
      infer:
        source: https://api.blackhole.ai.innopolis.university/public-datasets/folder/infer.zip
        target: folder/to/extract/infer/
    # ================================== #
           
      # some datamodules require additional arguments
      # look for them in the documentation of each datamodule
      # arguments passed in the following way:
      arg1: value1  # here arg1 - name of the argument, value1 - value for the arg1
      arg2: value2
      # ... same for other datamodule arguments


* \* `task` refers to the problem type where this datamodule is used. `framework` refers to the framework type where
  this datamodule is used

### Section 2.2. Writing own datamodule

Datamodule is a class which has the following responsibilities:
1. creation of data loaders for each dataset type: train, test, val and infer.
2. dataset setting up(e.g. downloading, preprocessing, creating additional files etc.) 
3. model predictions saving - formatting the predictions provided by a model



For now all of our data modules inherit from following two classes: `PandasDataModule`, `BaseLightningDataModule`

[//]: # (features of each datamodule)
PandasDataModule is suitable for tasks with input provided as table. The class provides the data by first uploading it into RAM.
BaseLightningDataModule is suitable for tasks where notion of 'batches' is reasonable for the data and the model.

```
from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)


class DataModule(BaseLightningDataModule):
   def setup(self, *args, **kwargs):
      pass
      
   def train_dataloader(self):
      pass

   def val_dataloader(self):
      pass

   def test_dataloader(self):
      pass
```

Where each dataloader utilizes the dataset(similar term as
[torch's Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html))

## Section 3. How to train a new model?

### Pytorch model

[//]: # (train a third-party model)

[//]: # (link in the conf file)


[//]: # (train your own model)
If you have written your own model, for instance this dummy model:

```python
import torch.nn as nn


class MNISTClassifier(nn.Module):
    def __init__(self, hidden_dim: int = 100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        return self.layers(x)
```

And you would like to add train it. Then you should do the following:

1. add `task` and `framework` parameters
    ```python
    import torch.nn as nn
    
   
   class MNISTClassifier(nn.Module):
        task = ['image-classification']
        framework = ['torch']
        # rest of the code is the same
    ```

   **standard list of tasks:**
    - image-classification
    - image-segmentation
    - image-detection
    - table-regression
    - table-classification
    - table-clustering
      ...

   **standard list of frameworks:**
    - torch
    - sklearn
    - xgboost


2. add the file with model to `innofw/core/models/torch/architectures/[task]/file_with_nn_module.py`
3. make sure dictionary in `get_default` in `innofw/utils/defaults.py` contains a mapping between your
   task and a lightning module

   if `task` has no corresponding `pytorch_lightning.LightningModule` add new implementation in this
   folder `innofw/core/models/torch/lightning_modules/[task].py`.

   > for more information on lightning modules
   visit [official documentation](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html)

4. make sure you have suitable dataset class for your model. Refer to [chapter 2]()

5. add configuration file to your model.

   in `config/models/[model_name].yaml` define a `_target_` field and arguments for your model.

   For example:

    ```yaml
    _target_: innofw.core.models.torch.architectures.classification.MNISTClassifier
    hidden_dim: 256
    ```

Now you are able to train and test your model! 😊

## Section 4. Start training, testing and inference

1. Make sure you have needed working dataset configuration. See Section 2.
2. Make sure you have needed working model configuration file. See Section 3.
3. Write an experiment file

   For instance file in folder `config/experiments` named `KA_130722_yolov5.yaml` with contents:
   ```yaml
   
   # @package _global_
   defaults:
     - override /models: [model_config_name]
     - override /datasets: [dataset_config_name]

   project: [project_name]
   task: [task_name]
   seed: 42
   epochs: 300
   batch_size: 4
   weights_path: /path/to/store/weights
   weights_freq: 1  # weights saving frequency

   ckpt_path: /path/to/saved/model/weights.pt
   ```
4. Launch training

```shell
python train.py experiments=KA_130722_yolov5.yaml
```

5. Launch testing

```shell
python test.py experiments=KA_130722_yolov5.yaml
```

6. Launch inference

```shell
python infer.py experiments=KA_130722_yolov5.yaml
```

## Section 5. Training on GPU

[//]: # (#devices:)

[//]: # ()

[//]: # (# - 0)

[//]: # ()

[//]: # (#gpus:)

[//]: # ()

[//]: # (# - 0 # 1)

[//]: # ()

[//]: # (# - 2 # 3)

[//]: # ()

[//]: # (# accelerator: cpu # цпу)

[//]: # ()

[//]: # (# accelerator: gpu  #)

[//]: # ()

[//]: # (#gpus: -1 # все гпу)

[//]: # ()

[//]: # (#)

[//]: # ()

[//]: # (#)

[//]: # ()

[//]: # (#accelerator: gpu)

[//]: # (#devices: 2 # найдет свободные гпу)

[//]: # (#accelerator: gpu # gpu, cpu)

[//]: # (#devices: 2 # If the devices flag is not defined,)

[//]: # ()

[//]: # (# it will assume devices to be "auto" and fetch the auto_device_count from the accelerator. [1])

[//]: # ()

[//]: # (#auto_select_gpus: True # False # [1])

[//]: # ()

References:

1. https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html

[//]: # ()

[//]: # (# 2.)

[//]: # (## What is the difference between a wrapper and an adapter?)

[//]: # ()

[//]: # (case: Wrapper)

[//]: # (cfg -> MODEL -> Wrapper -> wrapped_model .fit)

[//]: # (.test)

[//]: # (.val)

[//]: # (.predict # under the hood calls)

[//]: # ()

[//]: # (case: Adapter)

[//]: # (cfg -> adapted_model .fit # under the hood calls train function of underlying library/framework)


[//]: # (.test # under the hood calls train function of underlying library/framework)

[//]: # (.val # under the hood calls train function of underlying library/framework)

[//]: # (.predict # under the hood calls train function of underlying library/framework)

## Section 6. Versioning rules
1) Framework versions must be specified in X.Y.Z format, where:
‒ X – older version (updates in case of big changes);
‒ Y – younger version (updates in case of small changes);
‒ Z - tiny changes (updates in case of tiny changes).
2) When one of the numbers is increased, all numbers after it must be set to zero.
3) Backward compatibility in software must be maintained in all versions with the same older version.


