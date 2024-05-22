from innofw.core.callbacks.xgboost_callbacks.log_trainig_steps import XGBoostTrainingTensorBoardCallback

def test_xgboost_callback():
    cb = XGBoostTrainingTensorBoardCallback('./logs/test/test1/')
    assert cb is not None