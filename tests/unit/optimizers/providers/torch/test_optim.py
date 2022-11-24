from torch.optim import SGD, Adam, NAdam
import pytest

#
from innofw.core.optimizers import Optimizer
from tests.fixtures.models.torch.dummy_model import DummyTorchModel

model = DummyTorchModel()


@pytest.mark.parametrize(["optim"], [[SGD], [Adam], [NAdam]])
def test_torch_optim(optim):
    optimizer = Optimizer(optim(model.parameters(), lr=0.01))

    assert optimizer


