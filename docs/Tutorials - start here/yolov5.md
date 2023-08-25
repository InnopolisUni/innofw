# How to Use Innofw Framework for Object Detection using YoloV5

In this tutorial, we will show you how to use Innofw framework for object detection using YoloV5. We will show you how to create a configuration file for your dataset, how to create a configuration file for your training, how to train your model, how to evaluate your model and how to make an inference on your model.

There are different variations of yolov5 model. For example, you can use yolov5s, yolov5m, yolov5l or yolov5x. Models for YOLOv5 follow a certain order: 's', 'm', 'l', 'x'. This sequence signifies a progression from the smallest and fastest model to the largest and slowest one. It's important to note that while larger models tend to offer higher accuracy, they also demand more computational resources and take longer to process. You can change the model by changing the `arch` parameter in the file `config\models\detection\yolov5.yaml`.

## Dataset
* It's necessary for you to generate a configuration file tailored for your dataset. Please save this file in the following directory: `config\datasets\detection`. Within this directory, you can access existing configuration files that may serve as practical templates for your own. Take advantage of these resources to create an optimized configuration for your dataset.

* The configuration file should contain the following :
    1- Link to your dataset (if it is hosted on s3) or the path to your dataset (if it is on local machine). You will need to give the link to the training set, testing set and inference set separatly
    2- You need to add some information about the dataset such as the name, description, markup info, the task you are working on

* If your dataset needs a specific preprocessing, You can create your own adapter. After creating the adapter, you will need to add it to the configuration file of your dataset using `_target_` key.

## Training
* You need to create a configuration file for your experiment and save it inside the following directory: `config\experiments\detection`. You can see the current configuration files that currently exist in the directory. You can use them as a template for your training. You can also use the configuration file `template.yaml` as a template for your training.

* In the configuration file you can specify several parameters such as the following:
    1- The dataset
    2- The model
    3- The optimizer
    4- The scheduler
    5- The number of epochs
    6- The batch size
    7- The loss function
    8- The augmentation
    9- The accelerator
    10- The image size
    11- The number of workers


* Selecting the image size is very important. The bigger the image size, the more accurate the model is. However, the bigger the image size, the slower the training is. You can change the image size by changing the `img_size`.
* You can watch the gpu usage by running the following command: `watch -n 1 nvidia-smi`. You can also watch the cpu usage by running the following command: `htop`.
* After creating the configuration file, you can run the following command to start the training: `python train.py experiments= <path to your configuration file>`
* You can run more than one experiment at the same time.

## Checkpoints

* To use a checkpoint, you need to add the following parameters to your configuration file: `ckpt_path: <path to the directory of the checkpoint>` 

Example: `ckpt_path: logs\experiments\detection\yolov5s\2021-09-27\14-44-32\checkpoints\epoch=1-step=249.ckpt`

* You can also use the checkpoint path as a parameter to the command line. To do so, you need to add the following parameter to your command line: `ckpt_path= <path to the directory of the checkpoint>`

* If your checkpoint is uploaded on s3, you can add the checkpoint link to your checkpoint.

Example: `ckpt_path: https://api.blackhole.ai.innopolis.university/pretrained/testing/artaxor.pt`


## Evaluation 

* After training the model, you can use the checkpoint that was created by train.py to evaluate your model. You will find it inside logs folder

* You can evaluate your model using the following command: `python test.py experiments= <path to your configuration file> ckpt_path= <path to the directory of the checkpoint>`

* You don't need to specify the checkpoint path in the command line if you already specified it in the configuration file.

## Inference

* After training the model, you can use the checkpoint that was created by train.py to make an inference on your model. You will find it inside logs folder

* You can make an inference on your model using the following command: `python inference.py experiments= <path to your configuration file> ckpt_path= <path to the directory of the checkpoint>`

* You don't need to specify the checkpoint path in the command line if you already specified it in the configuration file.