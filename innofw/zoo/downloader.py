# standard libraries
from pathlib import Path
from typing import Optional

# third-party libraries
from fire import Fire
from urlpath import URL
from pydantic import validate_arguments, AnyUrl

from innofw.utils.executors.execute_w_creds import execute_w_credentials
from innofw.utils.s3_utils import S3Handler

# local modules
from innofw.utils.s3_utils import S3Handler
from innofw.constants import DefaultS3User, S3Credentials
from innofw.utils.s3_utils.credentials import get_s3_credentials


@execute_w_credentials
def _download_model(file_url, dst_path, server_url, credentials):
    return S3Handler(server_url, credentials).download_file(file_url, dst_path)


@validate_arguments
def download_model(
    file_url: AnyUrl,
    dst_path: Path,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> Path:
    """Function to download a model

    Arguments:
        file_url - url to the file
        dst_path - path where file should be stored. Can be relative to the project folder path.
         Possible to rename file by specifying new filename.
            Examples:
                1. pretrained/best_osl.ckpt
                2. pretrained/
                3. /home/user/innofw/pretrained/
        access_key - access key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials
        secret_key - secret key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials

    Usage:

    >>> from innofw.zoo import download_model

    >>> download_model(
    ...     file_url="https://api.blackhole.ai.innopolis.university/pretrained/model.pickle",
    ...     dst_path="pretrained/model.pickle",
    ...     access_key="some key",
    ...     secret_key="some secret"
    ...     )


    in cli:
        python innofw/zoo/downloader.py --file_url https://api.blackhole.ai.innopolis.university/pretrained/model.pickle\
                                        --dst_path pretrained/model.pickle\
                                        --access_key $access_key\
                                        --secret_key $secret_key
    """
    server_url = URL(file_url).anchor

    return _download_model(file_url=file_url, dst_path=dst_path, server_url=server_url)


if __name__ == "__main__":
    Fire(download_model)
