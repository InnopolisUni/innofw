# other
import hydra
import pytest
import torch
import torch.optim
from omegaconf import DictConfig
from segmentation_models_pytorch import Unet

from innofw.constants import Frameworks
from innofw.utils.framework import get_optimizer

# local


def test_optimizer_creation():
    cfg = DictConfig(
        {
            "optimizers": {
                "_target_": "innofw.core.optimizers.custom_optimizers.optimizers.LION",
                "lr": 1e-4,
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.torch
    model = Unet()
    optim_cfg = get_optimizer(cfg, "optimizers", task, framework)
    optim = hydra.utils.instantiate(optim_cfg, params=model.parameters())
    assert optim is not None and isinstance(optim, torch.optim.Optimizer)


def test_optimizer_creation_wrong_framework():
    cfg = DictConfig(
        {
            "optimizers": {
                "_target_": "innofw.core.optimizers.custom_optimizers.optimizers.LION",
                "lr": 1e-4,
            }
        }
    )
    task = "image-segmentation"
    framework = Frameworks.sklearn
    model = Unet()

    with pytest.raises(ValueError):
        optim = get_optimizer(
            cfg, "optimizers", task, framework, params=model.parameters()
        )
from innofw.core.optimizers.custom_optimizers.optimizers import LION

@pytest.fixture
def simple_model():
    # Define a simple model with one parameter
    torch.manual_seed(42)
    return torch.tensor([1.0], requires_grad=True)

def test_step_function(simple_model):
    # Define optimizer parameters
    lr = 0.1
    b1 = 0.9
    b2 = 0.99
    wd = 0.0

    # Create an instance of the _Lion optimizer
    optimizer = LION._Lion([simple_model], lr=lr, b1=b1, b2=b2, wd=wd)

    # Define a simple loss function
    loss_fn = torch.nn.MSELoss()

    # Perform a single optimization step
    def closure():
        output = simple_model * 2  # Some simple model
        loss = loss_fn(output, torch.tensor([5.0]))
        loss.backward()
        return loss

    loss_before = closure()
    optimizer.step(closure)
    loss_after = closure()

    # Check if the loss has decreased after optimization
    assert loss_after < loss_before

    # Check if the model parameter has been updated
    assert not torch.all(torch.eq(simple_model, torch.tensor([1.0])))
