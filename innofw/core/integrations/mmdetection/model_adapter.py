import os
import logging
from pathlib import Path
from typing import Optional

import torch
import yaml
from yolov5 import detect as yolov5_detect
from yolov5 import train as yolov5_train
from yolov5 import val as yolov5_val

from ..base_integration_models import BaseIntegrationModel
from innofw.constants import Frameworks
from innofw.core.models import BaseModelAdapter
from innofw.core.models import register_models_adapter
from innofw.utils.checkpoint_utils import TorchCheckpointHandler


class BaseMmdetModel(BaseIntegrationModel):
    framework = Frameworks.mmdetection

    def __init__(self, *args, **kwargs):
        pass


@register_models_adapter('mmdetection_adapter')
class Mmdetection3DDataModel(BaseModelAdapter):

    @staticmethod
    def is_suitable_model(model) -> bool:
        return isinstance(model, BaseMmdetModel)

    def _test(self, data):
        pass

    def _train(self, data):
        pass

    def _predict(self, data):
        pass

    framework = Frameworks.mmdetection

    def __init__(
            self,
            model,
            log_dir,
            trainer_cfg,
            *args,
            **kwargs,
    ):
        super().__init__(model, log_dir)
        self.device, self.epochs = trainer_cfg["accelerator"], trainer_cfg["max_epochs"]
        self.devices = trainer_cfg.get('devices', [])
        self.log_dir = Path(log_dir)
        self.mmdet_path = setup_mmdetection()

        self.data_path = os.path.join(self.mmdet_path, 'configs/_base_/datasets/custom.py')
        self.train_path = os.path.join(self.mmdet_path, 'configs/pointpillars/pointpillars_hv_secfpn_8xb6_custom.py')

        self.train_draft = ''
        self.data_draft = ''

    def update_configs(self, processed_data_path: str):
        with open(self.data_path, 'r') as d:
            self.data_draft = d.read()
        with open(self.data_path, 'w') as d:
            print(self.data_draft.replace('DATA_ROOT', processed_data_path), file=d)
        # logging.info(self.data_draft.replace('DATA_ROOT', processed_data_path))

        with open(self.train_path, 'r') as tr:
            self.train_draft = tr.read()
        with open(self.train_path, 'w') as tr:
            print(self.train_draft.replace('MAX_EPOCHS', str(self.epochs)), file=tr)
        # logging.info(self.train_draft.replace('MAX_EPOCHS', str(self.epochs)))

    def rollback_configs(self):
        with open(self.data_path, 'w') as d:
            print(self.data_draft, file=d)

        with open(self.train_path, 'w') as tr:
            print(self.train_draft, file=tr)

    def update_checkpoints_path(self):
        try:
            (self.log_dir / "weights").rename(self.log_dir / "checkpoints")

            try:
                dst_path = list((self.log_dir / "checkpoints").iterdir())[0]
                logging.info(f"Saved a checkpoint at: {dst_path}")
            except:
                pass
        except Exception as e:
            pass

    def train(self, data, ckpt_path=None):
        data.setup()
        self.update_configs(os.path.abspath(data.state['save_path']))
        logging.info('Training')
        devices = [] if self.device == 'cpu' else self.devices
        os.system(
            f'cd {self.mmdet_path} && sudo -E env "PATH=$PATH" "CUDA_VISIBLE_DEVICES={devices}" python tools/train.py configs/pointpillars/pointpillars_hv_secfpn_8xb6_custom.py')
        self.rollback_configs()

    def test(self, data, ckpt_path=None):
        data.setup()
        logging.info('Testing')
        devices = [] if self.device == 'cpu' else self.devices
        os.system(
            f'cd {self.mmdet_path} && sudo -E env "PATH=$PATH" "CUDA_VISIBLE_DEVICES={devices}" python tools/test.py configs/pointpillars/pointpillars_hv_secfpn_8xb6_custom.py')
        self.rollback_configs()


def setup_mmdetection():
    if os.path.exists('../../mmdetection3d'):
        os.environ['MMDET_FOR_INNOFW'] = os.path.join(Path(os.getcwd()).parent.parent, 'mmdetection3d')
    if 'MMDET_FOR_INNOFW' not in os.environ:
        os.system(
            "cd .. && git clone https://github.com/BarzaH/mmdetection3d.git && cd mmdetection3d && git checkout innofw_mod")
        os.environ['MMDET_FOR_INNOFW'] = os.path.join(Path(os.getcwd()).parent.parent, 'mmdetection3d')
    logging.info(os.environ['MMDET_FOR_INNOFW'])

    return os.environ['MMDET_FOR_INNOFW']