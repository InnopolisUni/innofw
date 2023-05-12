# How to Use Innofw Framework for Object Detection using YoloV5

In this tutorial, we will show you how to use Innofw framework for object detection using YoloV5. We will show you how to create a configuration file for your dataset, how to create a configuration file for your training, how to train your model, how to evaluate your model and how to make an inference on your model.

There are different variations of yolov5 library. For example, you can use yolov5s, yolov5m, yolov5l or yolov5x. The letter at the end of the name indicates the size of the model. The bigger the letter, the bigger the model. The bigger the model, the more accurate it is. However, the bigger the model, the slower it is. In this tutorial, we will use yolov5s. You can change the model by changing the `arch` parameter in the file `innofw\config\models\detection\yolov5.yaml`.

## Dataset
* You need to create a configuration file for your dataset and save it inside the following directory: `innofw\config\datasets\detection`. You can see the current configuration files that currently exist in the directory. You can use them as a template for your dataset. 

* The configuration file should contain the following :
    1- Link to your dataset (if it is public) or the path to your dataset (if it is private). You will need to give the link to the training set, testing set and inference set separatly
    2- You need to add some information about the dataset such as the name, description, markup info, the task you are working on
    3- If your dataset needs a specific preprocessing, You will need to create your own LightningDataModule and save it inside the following directory: `innofw\core\datamodules\lightning_datamodules\detection`. After creating the class, you will need to add it to the configuration file of your dataset using `_target_` key.

## Training
* You need to create a configuration file for your experiment and save it inside the following directory: `innofw\config\experiments\detection`. You can see the current configuration files that currently exist in the directory. You can use them as a template for your training. You can also use the configuration file `template.yaml` as a template for your training.

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
    10- The image size you want to use
    11- The number of workers you want to use


* Selecting the image size is very important. The bigger the image size, the more accurate the model is. However, the bigger the image size, the slower the training is. You can change the image size by changing the `img_size`.
* If you have enough cores, the number of workers is recommended to be 2 * batch_size. For example, if your batch size is 32, then the number of workers should be 64. You can change the number of workers by changing the `num_workers` parameter.
* You can watch the gpu usage by running the following command: `watch -n 1 nvidia-smi`. You can also watch the cpu usage by running the following command: `htop`.
* After creating the configuration file, you can run the following command to start the training: `python train.py experiments= <path to your configuration file>`
* You can run more than one experiment at the same time.

## Checkpoints

* To use a checkpoint, you need to add the following parameters to your configuration file: `ckpt_path: <path to the directory of the checkpoint>` 

## Evaluation 

* You can evaluate your model using the following command: `python test.py experiments= <path to your configuration file> ckpt_path= <path to the directory of the checkpoint>`

* If you already trained the model using train.py, you can use the checkpoint that was created by train.py to evaluate your model. You will find it inside logs folder

## Inference

* You can make an inference on your model using the following command: `python inference.py experiments= <path to your configuration file> ckpt_path= <path to the directory of the checkpoint>`

* If you already trained the model using train.py, you can use the checkpoint that was created by train.py to evaluate your model. You will find it inside logs folder