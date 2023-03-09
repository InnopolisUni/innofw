import math
from pathlib import Path

import albumentяations as albu
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from omegaconf import DictConfig

from innofw.utils import find_suitable_datamodule
from ui.utils import load_augmentations_config
from ui.utils import select_transformations
from ui.visuals import (
    get_transormations_params,
)

#
#


# init
if "conf_button_clicked" not in st.session_state:
    st.session_state.conf_button_clicked = False

if "new_batch_btn_clicked" not in st.session_state:
    st.session_state.new_batch_btn_clicked = False

if "indices" not in st.session_state:
    st.session_state.indices = None


def callback():
    st.session_state.conf_button_clicked = True


def callback_new_batch():
    st.session_state.new_batch_btn_clicked = True


if (
    st.sidebar.button("Configure!", on_click=callback)
    or st.session_state.conf_button_clicked
):
    task = st.sidebar.selectbox(
        "task", ["image-classification", "image-segmentation"]
    )
    framework = st.sidebar.selectbox("framework", ["torch", "sklearn"])

    data_path = st.sidebar.text_input("path to data")
    in_channels = st.sidebar.number_input(
        "input channels:", min_value=1, max_value=20
    )
    batch_size = st.sidebar.slider("batch size:", min_value=1, max_value=16)
    # prep_func_type = st.sidebar.selectbox('preprocessing function:', prep_funcs.keys())

    # /home/qazybek/Projects/InnoFramework3/tests/data/images/segmentation/arable

    # load the config
    augmentations = load_augmentations_config(
        None,
        str(Path("ui/augmentations.json").resolve()),  # placeholder_params
    )

    # get the list of transformations names
    interface_type = "Simple"
    transform_names = select_transformations(augmentations, interface_type)
    # get parameters for each transform
    transforms = [albu.Resize(300, 300, always_apply=True)]
    transforms = albu.ReplayCompose(transforms)

    # apply augmentations
    # augmentations = get_training_augmentation()

    dm_cfg = DictConfig({"task": [task], "data_path": data_path})
    if data_path is not None and data_path != "":
        dm = find_suitable_datamodule(task, framework, dm_cfg, aug=transforms)
        dm.setup()
        train_dataloader = dm.train_dataloader()

        if (
            "images" in st.session_state and "indices" in st.session_state
        ):  #  or ('preserve' in st.session_state and not st.session_state.preserve)
            images = st.session_state.images
            dataset_len = images.shape[0]
            with_replace = batch_size > dataset_len
            indices = st.session_state.indices
        else:
            batch = iter(train_dataloader).next()
            images = batch[0].detach().cpu().numpy()
            dataset_len = images.shape[0]
            with_replace = batch_size > dataset_len
            indices = np.random.choice(
                range(dataset_len), batch_size, with_replace
            )

        selected_aug = get_transormations_params(
            transform_names, augmentations
        )
        selected_aug = albu.ReplayCompose(selected_aug)

        aug_images = []
        for img in images:
            aug_img = selected_aug(image=img)["image"]
            aug_images.append(aug_img)

        ncols = 3 if batch_size >= 3 else batch_size
        nrows = math.ceil(batch_size / ncols)

        fig, axs = plt.subplots(nrows, ncols)
        if nrows > 1:
            axs = axs.flatten()
        elif ncols == 1:
            axs = [axs]

        # apply_aug = True
        # data = None

        st.title("Original Images")
        for i, ax in zip(indices, axs):
            ax.set_axis_off()
            ax.imshow(images[i])
        st.pyplot(fig)

        fig, axs = plt.subplots(nrows, ncols)
        if nrows > 1:
            axs = axs.flatten()
        elif ncols == 1:
            axs = [axs]

        dataset_len = len(aug_images)
        with_replace = batch_size > dataset_len

        st.title("Augmented Images")

        for i, ax in zip(indices, axs):
            ax.set_axis_off()
            ax.imshow(aug_images[i])

        st.pyplot(fig)

        col1, col2 = st.columns(2)

        st.session_state.indices = indices
        st.session_state.images = images

        b1 = col1.button("Обновить")

        def clear_img_ind():
            try:
                del st.session_state.images
            except:
                pass

            try:
                del st.session_state.indices
            except:
                pass

        b2 = col2.button("Следующий набор", on_click=clear_img_ind)
