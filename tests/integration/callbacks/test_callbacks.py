from innofw.core.callbacks.lightning_callbacks.detection.log_predictions import LogPredictionsDetectionCallback


def test_LogPredictionsDetectionCallback():
    cb = LogPredictionsDetectionCallback()
    assert cb is not None