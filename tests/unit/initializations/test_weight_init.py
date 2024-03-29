from innofw.utils.weights_initializations import WeightInitializer

import torch
import pytest


@pytest.mark.parametrize("initialization_type", [
    {'_target_': 'torch.nn.init.kaiming_uniform_'},
    {'_target_': 'torch.nn.init.xavier_uniform_'}
    ])
def test_initialization(initialization_type):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=(7, 7)),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(64),
    )
    model[0].weight.data = torch.zeros_like(model[0].weight)
    initializer = WeightInitializer(initialization_type)
    initializer.init_weights(model)
    assert  not torch.allclose(model[0].weight, torch.zeros_like(model[0].weight))
