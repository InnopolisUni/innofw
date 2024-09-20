# standard libraries
import logging.config
from pathlib import Path
from typing import Optional

import patoolib
from fire import Fire
from pydantic import AnyUrl
from pydantic import validate_arguments
from urlpath import URL

from innofw.constants import DefaultS3User
from innofw.constants import S3Credentials
from innofw.utils import get_project_root
from innofw.utils.executors.execute_w_creds import execute_w_credentials
from innofw.utils.s3_utils import S3Handler

# third-party libraries
# local modules

logging.config.fileConfig(get_project_root() / "logging.conf")
LOGGER = logging.getLogger(__name__)


@validate_arguments
def download_dataset(
    folder_url: AnyUrl,
    dst_path: Path,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
):
    """Function to download a dataset

        Arguments:
            folder_url - url to the dataset folder
            dst_path - path where archive file should be stored. Can be relative to the project folder path.
             Possible to rename file by specifying new filename.
                Examples:
                    1. datasets/credit_cards
                    2. datasets/
                    3. /home/user/innofw/datasets/
            access_key - access key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials
            secret_key - secret key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials

        Usage:

        >>> from innofw.data_mart import download_dataset

        >>> download_dataset(
        ...     folder_url="https://api.blackhole.ai.innopolis.university/public-datasets/test_dataset/",
        ...     dst_path="./datasets/test_dataset",
        ...     access_key="some key",
        ...     secret_key="some secret"
        ...     )


        in cli:
            python innofw/data_mart/downloader.py
                --folder_url https://api.blackhole.ai.innopolis.university/public-datasets/test_dataset/\
                --dst_path ./datasets/test_dataset\
                --access_key "some_key"\
                --secret_key "some_secret"
        """

    for file in ["train.zip", "test.zip"]:
        download_archive(file_url=f"{folder_url}/{file}", dst_path=dst_path)


@execute_w_credentials
def download_archive(
    file_url,
    dst_path,
    credentials: S3Credentials = DefaultS3User,
):
    new_dst_path = Path(dst_path)  # / URL(file_url).parts[-1]
    new_dst_path.mkdir(exist_ok=True, parents=True)
    url = URL(file_url).anchor
    downloaded_file = S3Handler(url, credentials).download_file(file_url, dst_path)
    # uncompress the file
    LOGGER.warning("might be doing redundant decompression")
    patoolib.extract_archive(str(downloaded_file), outdir=dst_path, interactive=False, verbosity=-1)
    # downloaded_file.unlink()

    # __MACOSX
    inner_files = list(dst_path.iterdir())
    inner_files = list(
        filter(
            lambda x: x.name not in ["__MACOSX"] and x.suffix != ".zip",
            inner_files,
        )
    )[0]
    return inner_files


if __name__ == "__main__":
    Fire(download_dataset)
