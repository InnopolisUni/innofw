# Exploratory data analysis
# ML stages:
# 1. understand the business context
# 2. frame the data science problem
# 3. explore and understand data
# 4. establish baseline metrics and models
# 5. communicate results
# table
# artifacts
# experiments
# reports
# todo: add code formatter
# todo: I gotta learn vim keybindings
from pathlib import Path
from typing import List

import cv2
import rasterio as rio
from pydantic import FilePath
from pydantic import validate_arguments
from rasterio import logging
from tqdm import tqdm

import wandb

log = logging.getLogger()
log.setLevel(logging.ERROR)
# def read_tiff(path):
#     return

from innofw.core.datasets.semantic_segmentation.tiff_dataset import read_tif


def create_table_augmentations(image_files, mask_files, augmentations):
    """function to create table to visualize effect of augmentations"""

    # add few more stat data


@validate_arguments
def create_table(
    image_files: List[FilePath],
    mask_files: List[FilePath],
    class_labels={0: "background", 1: "not background"},
) -> wandb.Table:
    """creates table"""
    table: wandb.Table = wandb.Table(
        columns=[
            "filename",
            "images",
            "height",
            "width",
            "num_channels",
            "pixel_ratio",
            "img_max_value",
            "nodata",
            "crs",
            "transform",
        ]
    )

    assert len(image_files) == len(mask_files)

    for image_file, mask_file in tqdm(
        zip(image_files, mask_files), total=len(image_files)
    ):
        image = read_tif(image_file)  # HWC

        height, width, num_channels = image.shape
        img_max_value = image.max()

        mask = read_tif(mask_file)  # HW1

        # rescale images
        image_rescaled = cv2.resize(
            image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC
        )
        mask_rescaled = cv2.resize(
            mask, dsize=(224, 224), interpolation=cv2.INTER_CUBIC
        )

        img_meta = rio.open(image_file).meta

        wandb_image = wandb.Image(
            image_rescaled,  # image
            masks={
                "ground_truth": {
                    "mask_data": mask_rescaled.squeeze(),  # mask
                    "class_labels": class_labels,
                }
            },
        )

        pixel_ratio = mask.sum() / mask.size

        table.add_data(
            image_file.stem,
            wandb_image,
            height,
            width,
            num_channels,
            pixel_ratio,
            img_max_value,
            str(img_meta["nodata"]),
            str(img_meta["crs"]),
            str(img_meta["transform"]),  # todo: consider this more
        )
    return table


def create_n_upload_artifact(image_files, mask_files):
    run = wandb.init(
        project="experimenting_w_wandb", entity="qazybi", job_type="upload"
    )

    artifact = wandb.Artifact("some_artifact", type="raw_data")

    # create a table with some images
    table = create_table(image_files[:100], mask_files[:100])
    artifact.add(table, "eda_table")

    run.log_artifact(artifact)


root_path = Path(
    "/mnt/nvmestorage/qb/data_n_weights/linear-road-bin-seg-oftp/200323/processed/2023-03-21-2048-merged-with-old/train"
)

image_files = list((root_path / "images").iterdir())
mask_files = list((root_path / "masks").iterdir())

# import numpy as np
create_n_upload_artifact(image_files, mask_files)

# what if this work from the torch dataset?
# I just want to upload multiple datasets
# or maybe even from datamodules

# datamodule1 + datamodule2 -> concat datamodule -> wandb table
#

# ds1 = Dataset()
# ds2 = Dataset()


# [ ] try to create a table with the whole dataset
# better to use multi threading for reading the images

# filter out some images
# furthermore, I had a script somewhere which removes some more redundant images
# the script used cv2

# add more fields(later will be used in grouping):
#   1. filename(from which file the tile was retrieved)
#   2. parent file name(for region)
#   3. dataset version(for now I have two dataset versions)
#   4. [x] nonbackground pixel ratio (important)
#   5. [x] image height
#   6. [x] image width
#   7. [x] maximal value
#   8. [x] some geo information: pixel size, crs
#   9. include information about the urban and not urban
# maybe I could train a classifier for that regard(I guess not needed)
#   10. [x] number of channels
#   11. [x] store only resized images: maybe to 224x224


# create a report about findings

# I want some histograms

# create an artifact for train val split
# creating a table where I write which of the datasets the image belongs

# !!!!!!!!!!!! important issues !!!!!!!!!!!!!!!

# model is not training: fixed by going to dice loss

# metrics are going to NaN
# loss is going to NaN

# ! the dataset is bigger thus I need to run the training script on three gpus !

# serving should be added too
# inference on the s3 links should be added too


# [ ] prepare reprojected version of the images and upload as an artifact to wandb


# ------ additional things -----

# also I would like to experiment with dataloading
# I want to measure the speed at which the steps gets processed
# in fact this could be done via sweeps,
# where I specify:
# 1. the number of workers,
# 2. batch size and number of gpus to run
# 3. and track the time

# -------------------------------


# +++++++++++++++++ Section 2 +++++++++++++++
# error analysis


# after training a model run an inference and compute loss on each image
# upload the wandb artifact to the wandb
# also somehow include confidence pixel mask
# that will be helpful to find those images which have bad masks


# ================= Extreme Testing =====================
# 1. make an inference on random tiles from different datasets and see how the model will perform(Munir)
# then add more such samples to the dataset and do fine-tuning


# ============= what about synthetic data? =============
