# standard libraries
from pathlib import Path

import pytest

from innofw.constants import DefaultS3User
from innofw.zoo import download_model

# third-party libraries
# local modules


@pytest.mark.parametrize(
    ["file_url", "credentials"],
    [
        [
            "https://api.blackhole.ai.innopolis.university/pretrained/testing/best.pkl",
            DefaultS3User,
        ]
    ],
)
def test_download_model(file_url, tmp_path, credentials):
    downloaded_file: Path = download_model(
        file_url,
        tmp_path,
        credentials.ACCESS_KEY.get_secret_value(),
        credentials.SECRET_KEY.get_secret_value(),
    )

    assert downloaded_file.exists()
    assert downloaded_file.parent == tmp_path

    downloaded_file.unlink()

    dst_path = tmp_path / "other_name.cbm"
    downloaded_file: Path = download_model(
        file_url,
        dst_path,
        credentials.ACCESS_KEY.get_secret_value(),
        credentials.SECRET_KEY.get_secret_value(),
    )
    assert downloaded_file.exists()
    assert dst_path.exists()
