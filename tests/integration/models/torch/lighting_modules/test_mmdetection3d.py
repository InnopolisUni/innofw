import os
import shutil

from omegaconf import DictConfig

from innofw.constants import Stages
from innofw.core.integrations.mmdetection.datamodule import Mmdetection3DDataModuleAdapter
from innofw.core.integrations.mmdetection.model_adapter import Mmdetection3DDataModel, \
    BaseMmdetModel
from tests.fixtures.config import optimizers as fixt_optimizers
from tests.fixtures.config import trainers as fixt_trainers


def test_integration():
    data_path = {
        'source': 'https://api.blackhole.ai.innopolis.university/public-datasets/lep_3d_detection/test.zip',
        'target': './tmp'}
    datamodule = Mmdetection3DDataModuleAdapter(data_path, data_path, data_path, 4)
    datamodule.setup_train_test_val()
    datamodule.setup_infer()
    assert datamodule.predict_dataloader() is None
    assert datamodule.train_dataloader() is None
    assert datamodule.test_dataloader() is None
    assert datamodule.save_preds(None, Stages.test, '') is None

    optimizer_cfg = DictConfig(fixt_optimizers.adam_optim_w_target)
    os.makedirs('./tmp/logs')
    model = Mmdetection3DDataModel(BaseMmdetModel(), './tmp/logs',
                                   fixt_trainers.trainer_cfg_w_gpu_devices_0,
                                   optimizers_cfg=optimizer_cfg)
    model.train(datamodule)
    ckpt_path = model.mmdet_path + '/work_dirs/pointpillars_hv_secfpn_8xb6_custom/epoch_10.pth'
    model.test(datamodule, ckpt_path=ckpt_path)
    model.predict(datamodule, ckpt_path=ckpt_path)

    for i in range(3):
        try:
            shutil.rmtree('./tmp')
            break
        except:
            pass
