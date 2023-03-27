import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from innofw.core.models.torch.architectures.segmentation import SegFormer
from innofw.utils.framework import get_model
from tests.fixtures.config.trainers import base_trainer_on_cpu_cfg


@pytest.mark.parametrize(
    ["config", "inp", "output_shape"],
    [
        [
            DictConfig(
                {
                    "_target_": "innofw.core.models.torch.architectures.segmentation.SegFormer",
                    "description": "Segmentation model based on transformers",
                    "name": "SegFormer",
                    "num_channels": 3,
                    "num_labels": 2,
                }
            ),
            torch.from_numpy(np.random.rand(4, 3, 512, 512)).float(),
            (4, 2, 128, 128),
        ],
        [
            DictConfig(
                {
                    "description": "Segmentation model based on transformers",
                    "name": "SegFormer",
                    "num_channels": 3,
                    "num_labels": 2,
                    "retain_dim": True,
                }
            ),
            torch.from_numpy(np.random.rand(4, 3, 512, 512)).float(),
            (4, 2, 512, 512),
        ],
        [
            DictConfig(
                {
                    "description": "Segmentation model based on transformers",
                    "name": "SegFormer",
                    "num_channels": 1,
                    "num_labels": 5,
                }
            ),
            torch.from_numpy(np.random.rand(4, 1, 512, 512)).float(),
            (4, 5, 128, 128),
        ],
        [
            DictConfig(
                {
                    "description": "Segmentation model based on transformers",
                    "name": "SegFormer",
                    "num_channels": 3,
                    "num_labels": 150,
                    "retain_dim": True,
                }
            ),
            torch.from_numpy(np.random.rand(1, 3, 512, 512)).float(),
            (1, 150, 512, 512),
        ],
    ],
)
def test_inference(config, inp, output_shape):
    model = get_model(config, base_trainer_on_cpu_cfg)
    output = model(inp)
    assert output.shape == output_shape


@pytest.mark.parametrize(
    ["config", "inp"],
    [
        [
            DictConfig(
                {
                    "_target_": "innofw.core.models.torch.architectures.segmentation.SegFormer",
                    "description": "Segmentation model based on transformers",
                    "name": "SegFormer",
                    "num_channels": 1,
                    "num_labels": 10,
                }
            ),
            torch.from_numpy(np.random.rand(4, 3, 512, 512)).float(),
        ],
        [
            DictConfig(
                {
                    "description": "Segmentation model based on transformers",
                    "name": "SegFormer",
                    "num_channels": 3,
                    "num_labels": 2,
                }
            ),
            torch.from_numpy(np.random.rand(4, 3, 1, 1)).float(),
        ],
    ],
)
def test_incorrect_inference(config, inp):
    model = get_model(config, base_trainer_on_cpu_cfg)
    with pytest.raises(RuntimeError):
        model(inp)


@pytest.mark.parametrize(
    ["config"],
    [
        [
            DictConfig(
                {
                    "_target_": "innofw.core.models.torch.architectures.segmentation.SegFormer",
                    "name": "SegFormer",
                    "num_channels": 3,
                    "num_labels": 10,
                    "hidden_sizes": [16, 32, 64, 128, 256, 400],
                }
            ),
        ],
        [
            DictConfig(
                {
                    "name": "SegFormer",
                    "num_channels": 3,
                    "num_labels": 2,
                    "num_encoder_blocks": 5,
                }
            ),
        ],
    ],
)
def test_wrong_config(config):
    with pytest.raises(ValueError):
        get_model(config, base_trainer_on_cpu_cfg)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU is found on this machine"
)
def test_gpu_inference():
    device = torch.device("cuda")

    seg = SegFormer(
        num_labels=5,
        num_channels=3,
        reduce_labels=True,
        retain_dim=True,
    ).to(device)
    image = torch.from_numpy(np.random.rand(1, 3, 512, 512)).float().to(device)
    outputs = seg(image)
    assert outputs.size() == (1, 5, 512, 512)
