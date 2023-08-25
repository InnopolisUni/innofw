import pickle
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from innofw.zoo import download_model


@patch("innofw.utils.s3_utils.S3Handler.download_file")
@pytest.mark.parametrize(
    ["file_url"],
    [
        [
            "https://api.blackhole.ai.innopolis.university/pretrained/testing/best.pkl",
        ]
    ],
)
def test_download_model(mock_download_model, file_url, tmp_path):
    with open(tmp_path / "model.pkl", "wb+") as f:
        pickle.dump(torch.nn.Module(), f)
    mock_download_model.return_value = tmp_path / "model.pkl"
    downloaded_file: Path = download_model(
        file_url,
        tmp_path,
    )

    assert downloaded_file.exists()
    assert downloaded_file.parent == tmp_path

    downloaded_file.unlink()

    # Test case 1: Try to download with another name and check if the file is save with another name.
    with open(tmp_path / "other.pkl", "wb+") as f:
        pickle.dump(torch.nn.Module(), f)
    mock_download_model.return_value = tmp_path / "other.pkl"
    downloaded_file: Path = download_model(
        file_url,
        tmp_path,
    )
    assert downloaded_file.exists()
    assert downloaded_file.parent == tmp_path

    downloaded_file.unlink()
