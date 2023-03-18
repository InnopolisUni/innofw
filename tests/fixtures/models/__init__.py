#
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from tests.fixtures.models.torch.dummy_model import DummyLightningModel
from tests.fixtures.models.torch.dummy_model import DummyTorchModel

#


xgb_reg_model = XGBRegressor()
sklearn_reg_model = LinearRegression()
catboot_cls_model = CatBoostClassifier()
torch_model = DummyTorchModel(
    in_channels=28 * 28, out_channels=10
)  # the model is configured for the mnist classification problem
lightning_model = DummyLightningModel(torch_model)
