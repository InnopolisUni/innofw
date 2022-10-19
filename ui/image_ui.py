# def image_input_handler(st):
#     task = st.selectbox(
#         "What task you want to solve?",
#         [
#             "segmentation",
#             "detection",
#             "image generation",
#             "classification",
#             "regression",
#         ],
#     )
#     src_data_folderpath = st.text_input("Path to the folder with data")
#
#     datamodule_cfg = {}
#     model_params = {}
#     recreate_compressed_format = st.multiselect(
#         "Which datasets should be compressed/recompressed?",
#         ["fit", "val", "test"],
#         default=[],
#     )  # 'fit', 'val', 'test'
#
#     bands_names = st.multiselect(
#         "Image channel names",
#         ["RED", "GRN", "BLU", "NIR"],
#         default=["RED", "GRN", "BLU", "NIR"],
#     )
#
#     compressed_ds_dst_path = st.text_input("Where to store compressed files?")
#
#     datamodule_cfg = {
#         "overwrite": recreate_compressed_format,
#         "src_data_folderpath": make_valid_path(src_data_folderpath),
#         "dst_tile_ds_path": make_valid_path(compressed_ds_dst_path),
#         "bands_idx": bands_names,
#     }
#     model = None
#     if task == "segmentation":
#         model = st.selectbox(
#             "What model you want to try?", ["Unet", "Unet++", "DeepLabV3", "DeepLabV3+"]
#         )
#         model_params["name"] = model.lower()
#
#         model_size = st.selectbox(
#             "What size of the model you want?", ["Small", "Medium", "Large"]
#         )
#
#         with st.expander("Open to tune parameters"):
#             model_params["activation"] = st.selectbox(
#                 "activation function", ["sigmoid", "None", "softmax"]
#             )
#
#             if model_size == "Small":
#                 model_params["encoder_name"] = st.selectbox(
#                     "encoder", ["resnet18", "dpn68"]
#                 )
#                 # model_params['n_jobs'] = st.slider("n_jobs", value=1, min_value=-1, max_value=16)
#             elif model_size == "Medium":
#                 model_params["encoder_name"] = st.selectbox(
#                     "encoder", ["resnet34", "resnet50"]
#                 )
#             elif model_size == "Large":
#                 model_params["encoder_name"] = st.selectbox(
#                     "encoder", ["resnet101", "resnet152"]
#                 )
#             else:
#                 raise NotImplementedError
#     elif task == "classification":
#         # get the number of classes
#         model = st.selectbox(
#             "Выберите модель:",
#             ["resnet18", "alexnet", "vgg16", "googlenet", "inception_v3"],
#         )
#         model_params["name"] = model.lower()
#
#     model_params["classes"] = st.number_input("number of classes", value=1, min_value=1)
#     model_params["in_channels"] = st.number_input(
#         "number of channels", value=3, min_value=1, max_value=8
#     )
#
#     if model is not None:
#         # dictionary where key is the filename where values are the descriptions
#         suitable_metrics = {}
#         metrics_files = list(Path("config/metrics/").iterdir())
#         from omegaconf import OmegaConf
#         from innofw.utils.framework import (
#             is_suitable_for_framework,
#             is_suitable_for_task,
#             map_model_to_framework,
#         )
#         from innofw import utils
#
#         name = model_params["name"]
#         model_params["_target_"] = utils.find_suitable_model(name)
#         del model_params["name"]
#         if task == "segmentation":
#             model = hydra.utils.instantiate(model_params)
#         elif task == "classification":
#             model_params_copy = model_params.copy()
#             del model_params_copy["in_channels"]
#             del model_params_copy["classes"]
#             # del model_params_copy['in_channels']
#             model = hydra.utils.instantiate(model_params_copy)
#         model_params["name"] = name
#         framework = map_model_to_framework(model)
#         for file in metrics_files:
#             with open(file, "r") as f:
#                 contents = OmegaConf.load(f)
#                 try:
#                     # st.info(f"image-{task}" in {contents.requirements.task})
#                     if is_suitable_for_task(
#                         contents, f"image-{task}"
#                     ) and is_suitable_for_framework(contents, framework):
#                         suitable_metrics[file.stem] = contents.objects[
#                             framework
#                         ].description
#                 except:
#                     pass
#
#         if len(suitable_metrics) == 0:
#             st.warning(
#                 f"Unable to find required metrics for model: {model_params['name']}"
#             )
#         else:
#             metrics = st.multiselect("What metrics to measure", suitable_metrics.keys())
#             if len(metrics) != 0:
#                 st.write(f"Metrics Description: {suitable_metrics[metrics[0]]}")
#
#     # ====== Configuration Creation ===== #
#     if st.button("Create!"):
#         save_config("image", task, datamodule_cfg, model_params)
#         st.markdown(
#             "[click here for augmentations](/Аугментация)", unsafe_allow_html=True
#         )
#         # import streamlit.components.v1 as components
#         # components.html("<a href='Аугментация'>click here</a>")
#
