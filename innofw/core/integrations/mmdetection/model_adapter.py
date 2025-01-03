import os
import logging
import sys
from pathlib import Path
import subprocess
from typing import Optional

import torch
import yaml

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
        self.optimizer = map_optimizer_to_mmdet_optim(kwargs['optimizers_cfg']['_target_'].split('.')[-1].lower())
        self.optim_lr = kwargs['optimizers_cfg']['lr']
        self.device, self.epochs = trainer_cfg["accelerator"], trainer_cfg["max_epochs"]
        self.devices = trainer_cfg.get('devices', [])
        self.log_dir = Path(log_dir)
        self.mmdet_path = setup_mmdetection()

        self.data_path = os.path.join(self.mmdet_path, 'configs/_base_/datasets/newcustom.py')
        self.train_path = os.path.join(self.mmdet_path, 'configs/centerpoint/centerpoint_baseline_custom_bs2.py')

        self.train_draft = ''
        self.data_draft = ''

    def update_configs(self, processed_data_path: str):
        with open(self.data_path, 'r') as d:
            self.data_draft = d.read()
        with open(self.data_path, 'w') as d:
            print(self.data_draft.replace('DATA_ROOT', processed_data_path), file=d)

        with open(self.train_path, 'r') as tr:
            self.train_draft = tr.read()
        with open(self.train_path, 'w') as tr:
            code = self.train_draft.replace('MAX_EPOCHS', str(self.epochs))
            code = code.replace('OPTIM_TYPE', self.optimizer)
            code = code.replace('OPTIM_LR', str(self.optim_lr))
            print(code, file=tr)

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
        logging.info('Training')

        devices = [] if self.device == 'cpu' else self.devices



        run_env = os.environ.copy()
        run_env["NO_CLI"] = "True"
        run_env["PYTHONPATH"] = "."
        run_env["CUDA_VISIBLE_DEVICES"] = f"{devices}"

        # os.system(f'cd {self.mmdet_path}')
        cmd = [sys.executable, "tools/train.py",
               "configs/centerpoint/centerpoint_baseline_custom_bs2.py",
               f"--work-dir={self.log_dir}",
               f"--data_root={os.path.abspath(data.state['save_path'])}",
               f"--class_names={data.class_names}",
               f"--max_epochs={self.epochs}"]

        if ckpt_path:
            cmd.append(f"--resume={ckpt_path}")

        try:
            sp = subprocess.Popen(cmd, cwd=self.mmdet_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=run_env)
            output = ""
            while True:
                # Read line from stdout, break if EOF reached, append line to output
                line = sp.stdout.readline()
                line = line.decode()
                if line == "":
                    break
                print(line)
                output += line
        except Exception as e:
            print(e)


    def test(self, data, ckpt_path=None, flags=''):
        devices = [] if self.device == 'cpu' else self.devices
        try:
            os.system(
                f'cd {self.mmdet_path} && sudo -E env "PATH=$PATH" "PYTHONPATH=." "CUDA_VISIBLE_DEVICES={devices}" {sys.executable} tools/test.py configs/pointpillars/pointpillars_hv_secfpn_8xb6_custom.py {ckpt_path} {flags}')
        except:
            logging.info('Failed')

    def predict(self, data, ckpt_path=None):
        devices = [] if self.device == 'cpu' else self.devices

        run_env = os.environ.copy()
        run_env["NO_CLI"] = "True"
        run_env["PYTHONPATH"] = "."
        run_env["CUDA_VISIBLE_DEVICES"] = f"{devices}"

        cmd = [sys.executable,
               "tools/infer2.py",
               "configs/centerpoint/centerpoint_baseline_custom_bs2.py",
               ckpt_path,
               data.state["data_path"],
               self.log_dir,
               f"--data_root={os.path.abspath(data.state['save_path'])}",
               f"--class_names={data.class_names}"]
        try:
            sp = subprocess.Popen(cmd, cwd=self.mmdet_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  env=run_env)
            output = ""
            while True:
                # Read line from stdout, break if EOF reached, append line to output
                line = sp.stdout.readline()
                line = line.decode()
                if line == "":
                    break
                print(line)
                output += line
        except:
            logging.info('Failed')


def setup_mmdetection():
    if os.path.exists('../mmdetection3d'):
        logging.info('mmdetection-3d found')
        os.environ['MMDET_FOR_INNOFW'] = os.path.join(Path(os.getcwd()).parent, 'mmdetection3d')
    if 'MMDET_FOR_INNOFW' not in os.environ:
        logging.info("Cloning mmdetection-3d")
        os.system(
            "cd .. && git clone https://github.com/BarzaH/mmdetection3d.git && cd mmdetection3d && git checkout innofw_centerpoint")
        os.environ['MMDET_FOR_INNOFW'] = os.path.join(Path(os.getcwd()).parent, 'mmdetection3d')
    logging.info("mmdetection-3d path " + os.environ['MMDET_FOR_INNOFW'])

    return os.environ['MMDET_FOR_INNOFW']


def map_optimizer_to_mmdet_optim(optim_name):
    return {'adam': 'Adam', 'adamw': 'AdamW', 'sgd': 'SGD'}[optim_name]
