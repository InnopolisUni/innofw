if __name__ == "__main__":
    import math
    from pathlib import Path

    #
    import numpy as np
    import streamlit as st
    import albumentations as albu
    from omegaconf import DictConfig
    import matplotlib.pyplot as plt

    from innofw.constants import Frameworks, CLI_FLAGS
    from innofw.core.augmentations import Augmentation

    #
    from innofw.utils import find_suitable_datamodule
    from innofw.utils.framework import (
        get_datamodule,
        get_obj,
        get_model,
        map_model_to_framework,
    )
    from innofw.utils.getters import get_trainer_cfg
    from ui.utils import (
        load_augmentations_config,
        get_arguments,
        get_placeholder_params,
        select_transformations,
        show_random_params,
    )

    from ui.visuals import (
        select_image,
        show_credentials,
        show_docstring,
        get_transormations_params,
    )

    # read from configuration file
    # instantiate the augmentations
    # show a textarea with config file contents

    import os
    import sys
    import random
    import argparse
    import streamlit as st
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="This app provides UI for the framework"
    )

    parser.add_argument("experiments", default="ui.yaml", help="Config name")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)

    # set up the env flag
    try:
        os.environ[CLI_FLAGS.DISABLE.value] = "True"
    except:
        pass

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

    # if st.sidebar.button("Configure!", on_click=callback) or st.session_state.conf_button_clicked:

    st.header("LOGO")

    # task = st.sidebar.selectbox("task", ["image-classification", "image-segmentation"])
    # framework = st.sidebar.selectbox(
    #     "framework", list(Frameworks), format_func=lambda x: x.value.lower()
    # )

    # data_path = st.sidebar.text_input("path to data")
    # in_channels = st.sidebar.number_input("input channels:", min_value=1, max_value=20)
    # batch_size = st.sidebar.slider("batch size:", value=3, min_value=1, max_value=16)
    # prep_func_type = st.sidebar.selectbox('preprocessing function:', prep_funcs.keys())

    # get the list of transformations names
    interface_type = "Simple"  # todo:
    # transform_names = select_transformations(augmentations, interface_type)
    # get parameters for each transform
    transforms = [albu.Resize(300, 300, always_apply=True)]
    transforms = albu.ReplayCompose(transforms)

    # apply augmentations
    # augmentations = get_training_augmentation()
    from hydra import compose, initialize
    import hydra

    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    selected_aug = None
    aug_conf = None

    # from hydra.core.global_hydra import GlobalHydra
    # GlobalHydra.instance().clear()

    # load the config
    augmentations = load_augmentations_config(
        None, str(Path("ui/augmentations.json").resolve())  # placeholder_params
    )

    try:
        initialize(config_path="../../config/", version_base="1.1")
    except:
        pass
    cfg = compose(
        config_name="train",
        overrides=[f"experiments={args.experiments}"],
        return_hydra_config=True,
    )

    dm_cfg = cfg["datasets"]
    # dm_cfg = DictConfig({
    #     'task': [task],
    #     'data_path': data_path
    # })
    if dm_cfg["train"]["source"] is not None and dm_cfg["train"]["source"] != "":
        trainer_cfg = get_trainer_cfg(cfg)
        model = get_model(cfg.models, trainer_cfg)
        task = cfg.get("task")
        framework = map_model_to_framework(model)
        batch_size = cfg.get("batch_size")
        if batch_size is None:
            batch_size = 6
        # st.info(f"creating datamodule")

        if "train_dataloader" in st.session_state:
            train_dataloader = st.session_state.train_dataloader
        else:
            dm = get_datamodule(dm_cfg, framework, task=task, augmentations=transforms)

            dm.setup()

            train_dataloader = dm.train_dataloader()
            st.session_state.train_dataloader = train_dataloader

        # st.info(f"getting images")
        if (
            "images" in st.session_state and "indices" in st.session_state
        ):  # or ('preserve' in st.session_state and not st.session_state.preserve)
            images = st.session_state.images
            dataset_len = images.shape[0]
            with_replace = batch_size > dataset_len
            indices = st.session_state.indices
        else:
            batch = iter(train_dataloader).next()
            try:
                images = batch[0].detach().cpu().numpy()
                # images = batch["scenes"].detach().cpu().numpy()
            except:
                if isinstance(batch, list):
                    images = (
                        batch[0].detach().cpu().numpy()
                    )  # np.array([img.detach().cpu().numpy() for img in batch])
                else:
                    images = batch.detach().cpu().numpy()

            dataset_len = images.shape[0]
            with_replace = batch_size > dataset_len
            indices = np.random.choice(range(dataset_len), batch_size, with_replace)

        if not isinstance(images, np.ndarray):
            images = images.detach().cpu().numpy()

        # selected_aug = None

        try:
            # st.info("initializing augmentations")
            selected_aug = Augmentation(get_obj(cfg, "augmentations", task, framework))
            aug_conf = cfg["augmentations"]["implementations"]["torch"]["Compose"][
                "object"
            ]
            # selected_aug = hydra.utils.instantiate(aug_conf)
        except Exception as e:
            st.warning(f"unable to process the train.yaml file. {e}")

        # st.info(f"{augmentations} {aug_conf}")

        if selected_aug is not None and aug_conf is not None:
            # st.info("created augmentations")
            st.info("Результаты применения трансформации")

            # selected_aug = get_transormations_params(transform_names, augmentations)
            # selected_aug = albu.ReplayCompose(selected_aug)

            aug_images = []
            for img in images:
                try:
                    aug_img = selected_aug(img)["image"]
                    aug_img = aug_img[:3, ...]
                except:
                    aug_img = selected_aug(img)
                    aug_img = aug_img[:3, ...]

                if not isinstance(aug_img, np.ndarray):
                    aug_img = aug_img.detach().cpu().numpy()

                aug_img = np.moveaxis(aug_img, 0, -1)
                aug_images.append(aug_img)

            ncols = 3 if batch_size >= 3 else batch_size
            nrows = math.ceil(batch_size / ncols)

            fig, axs = plt.subplots(nrows, ncols)
            if nrows > 1:
                axs = axs.flatten()
            elif ncols == 1:
                axs = [axs]

            # apply_aug = True  # todo: retrieve from user input
            # data = None

            st.warning("Исходные изображения")
            for i, ax in zip(indices, axs):
                ax.set_axis_off()
                imgs = images[i]
                # st.info(f"orig: {imgs.shape}")

                # imgs = np.moveaxis(imgs, 0, -1)  # [..., :3]

                ax.imshow(imgs)
            st.pyplot(fig)

            fig, axs = plt.subplots(nrows, ncols)
            if nrows > 1:
                axs = axs.flatten()
            elif ncols == 1:
                axs = [axs]

            dataset_len = len(aug_images)
            with_replace = batch_size > dataset_len

            st.warning("Применение трансформации")

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
            from omegaconf import OmegaConf
            from pprint import pformat

            formatted_conf_str = pformat(OmegaConf.to_yaml(aug_conf), indent=4)
            formatted_conf_str = [
                item.replace("\\n", "\n").replace("'", "")
                for item in formatted_conf_str[1:-1].split("\n")
            ]
            formatted_conf_str = "\n".join(formatted_conf_str)
            st.markdown("\n**Конфигурация:**\n")
            st.text(formatted_conf_str)
