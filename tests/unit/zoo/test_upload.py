import pickle
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch

from innofw.zoo import upload_model


@patch("innofw.utils.s3_utils.credentials.get_s3_credentials")
@patch("innofw.utils.s3_utils.S3Handler.upload_file")
@patch("builtins.input")
@pytest.mark.parametrize(
    ["remote_save_path", "experiment_config_path", "metrics"],
    [
        [
            "https://api.blackhole.ai.innopolis.university/pretrained/model.pickle",
            "classification/DS_190423_8dca23dc_ucmerced.yaml",
            {"some metric": 0.04},
        ]
    ],
)
def test_upload_model(
    mock_get_s3_credentials,
    mock_upload_file,
    mock_input,
    tmp_path,
    experiment_config_path,
    remote_save_path,
    metrics,
):
    ckpt_path = tmp_path / "model.pkl"
    with open(ckpt_path, "wb+") as f:
        pickle.dump(torch.nn.Module(), f)

    mock_input.side_effect = ["access_key", "secret_key"]
    mock_get_s3_credentials.return_value = MagicMock()
    mock_upload_file.return_value = remote_save_path

    # Call the function being tested
    upload_model(
        experiment_config_path,
        ckpt_path,
        remote_save_path,
        metrics,
    )

    # Assert that the expected functions were called with the expected arguments
    mock_upload_file.assert_called_once()


    # Test case 1: Try to upload a file that doesn't exist and check if it raises an exception.
    with pytest.raises(Exception):
        upload_model(
            experiment_config_path,
            "invalid",
            remote_save_path,
            metrics,
        )
