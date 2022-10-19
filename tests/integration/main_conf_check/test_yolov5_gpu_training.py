# from tests.fixtures.datasets import lep_datamodule_cfg_w_target
# from tests.fixtures.models import yolov5_cfg_w_target
# from tests.fixtures.trainers import trainer_cfg_w_gpu_devices_1
# from innofw.utils.framework import get_datamodule, get_model, map_model_to_framework
# from innofw.wrappers import Wrapper
#
#
# def test_on_gpu2():
#     model_cfg, dm_cfg, trainer_cfg, task = yolov5_cfg_w_target, lep_datamodule_cfg_w_target, trainer_cfg_w_gpu_devices_1, "image-detection",
#     # create a model
#     model = get_model(model_cfg, trainer_cfg)
#     # create a dataset
#     framework = map_model_to_framework(model)
#     dm = get_datamodule(dm_cfg, framework=framework)
#     # wrap a model
#     wrp_model = Wrapper.wrap(model, trainer_cfg=trainer_cfg, task=task)
#     # start training
#     wrp_model.train(dm)
#
#     assert True
