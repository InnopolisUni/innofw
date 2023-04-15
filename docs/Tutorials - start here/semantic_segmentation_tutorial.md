# How to Use Innofw Framework for Semantic Segmentation

## Dataset
* You need to create a configuration file for your dataset and save it inside the following directory: `innofw\config\datasets\semantic-segmentation`. You can see the current configuration files that currently exist in the directory. You can use them as a template for your dataset. 

* The configuration file should contain the following :
    1- Link to your dataset (if it is public) or the path to your dataset (if it is private). You will need to give the link to the training set, testing set and inference set separatly
    2- You need to add some information about the dataset such as the name, description, markup info, the task you are working on
    3- If your dataset needs a specific preprocessing, You will need to create your own LightningDataModule and save it inside the following directory: `innofw\core\datamodules\lightning_datamodules\semantic_segmentation`. After creating the class, you will need to add it to the configuration file of your dataset using `_target_` key.



## Training
* You need to create a configuration file for your training and save it inside the following directory: `innofw\config\experiments\semantic-segmentation`. You can see the current configuration files that currently exist in the directory. You can use them as a template for your training. You can also use the configuration file `template.yaml` as a template for your training.

* In the configuration file you can specify several parameters such as the following:
    1- The dataset you are using
    2- The model you want to use
    3- The optimizer you want to use
    4- The scheduler you want to use
    5- The number of epochs
    6- The batch size
    7- The loss function you want to use
    8- The augmentation you want to use
    9- The accelerator you want to use

* After creating the configuration file, you can run the following command to start the training: `python train.py experiments= <path to your configuration file>`
* You can run more than one experiment at the same time.

## Checkpoints

* 

## Evaluation 

* You can evaluate your model using the following command: `python test.py experiments= <path to your configuration file>`

## Inference

* You can make an inference on your model using the following command: `python inference.py experiments= <path to your configuration file>`

    