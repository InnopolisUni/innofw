[![Documentation Status](https://readthedocs.org/projects/innofw/badge/?version=latest)](https://innofw.readthedocs.io/en/latest/?badge=latest)

# What is InnoFW?

InnoFW is a configuration-based machine learning framework that helps people to get started with development of machine learning solutions. InnoFW is easy to pickup and play. And easy to master.

- define configuration files for models, datasets, optimizers, losses, metrics etc. and interchange them with one another
- have unified and intuitive code structure
- powerful CLI for argument passing
- train models on multiple gpu by passing a flag
- select loggers: Tensorboard, ClearML, Wandb and ...(upcoming)
- Easy work with S3-like storages


# Why to use InnoFW?

This framework serves as a template for different projects members to start working on a problem. Machine learning engineers can enjoy ease of integration of a new model and software developers can value unified and documented API for model training and inference.


> Please note that the project is under early development stage.


# Project is based on
1. pytorch lightning
2. hydra
3. pydantic
4. sklearn
5. torch


# How It Works
InnoFW uses hydra to provide configuration structure that is suitable for the most machine learning projects.

1. Create an experiment config file in the folder ```config/experiments/``` based on ```config/experiments/template.yaml```.
2. Once you define your configuration file you can start training your model.
    ```python train.py experiments=yolov5_cars```
3. InnoFW checks the configuration file for consistency of individual modules(model, dataset, loss, optimizer etc.) and if everything is fine then selects and adapter. Adapter is responsible for starting the training, testing, validation and inference pipeline.
4. Model is being trained and checkpoints saved.


# Quick start

1. install python 3.8-3.9 to your system
2. clone project
    ```git clone https://github.com/InnopolisUni/innofw.git```
3. create virtual env
    ```python -m venv venv```
4. install packages
    ```pip install -r requirements.txt```


For full guide on installation using poetry or docker follow this [quick start page](https://innofw.readthedocs.io/en/latest/quick-start)



# Covered Tasks:
- semantic segmentation
- image classification
- object detection
- tabular data regression
- tabular data classification
- tabular data clustering
- one-shot learning
- anomaly detection in time series


# Models List:
- yolov5(+seg)
- segmentation-models.pytorch models
- sklearn
- torchvision's resnet18
- lstm
- one-shot learning
- biobert


<!-- inspirations:
1. lightning flash
2. ludwig
3. catalyst -->
