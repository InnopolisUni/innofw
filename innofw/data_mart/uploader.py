# standard libraries
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List
from typing import Optional
from typing import Union

from fire import Fire
from pydantic import AnyUrl
from pydantic import validate_arguments
from pydantic.types import DirectoryPath
from urlpath import URL
import hydra
from omegaconf.errors import ConfigKeyError

from innofw.constants import S3Credentials
from innofw.constants import S3FileTags
from innofw.schema.dataset import DatasetConfig
from innofw.utils import get_abs_path
from innofw.utils.file_hash import compute_file_hash
from innofw.utils.s3_utils import S3Handler
from innofw.utils.s3_utils.credentials import get_s3_credentials

# third-party libraries
# local modules


def upload_dataset(
    dataset_config_path: str,
    remote_save_path: str,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    **kwargs,
):
    """Function to upload a dataset into s3(remote storage) and generate config file for the dataset

    Arguments:
        dataset_config_path: path to the dataset config (relatively to config/datasets folder)
        remote_save_path - url to the dataset save location.
            New filename can be specified for the archive file
            Example:
             1. https://api.blackhole.ai.innopolis.university/public-datasets/
             2. https://api.blackhole.ai.innopolis.university/public-datasets/credit_cards.zip

        access_key - access key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials
        secret_key - secret key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials

    Usage:

    >>> from innofw.data_mart import upload_dataset

    >>> upload_dataset(
    ...     dataset_config_path = "classification/industry_data",
    ...     remote_save_path = "https://api.blackhole.ai.innopolis.university/public-datasets/industry_data.zip",
    ...     access_key = "some key",
    ...     secret_key = "some secret"
    ...     )

    in cli:
        python innofw/data_mart/uploader.py --dataset_config_path classification/industry_data
                                            --remote_save_path https://api.blackhole.ai.innopolis.university/public-datasets/test_dataset/\
    """
    if access_key is None or secret_key is None:
        credentials = get_s3_credentials()
    else:
        credentials = S3Credentials(ACCESS_KEY=access_key, SECRET_KEY=secret_key)
    with hydra.initialize(config_path="../../config", version_base="1.2"):
        config = hydra.compose(
            config_name="train.yaml", overrides=[f"datasets={dataset_config_path}"]
        )

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for folder in ["train", "test"]:
            dst_filename = tmpdir / f"{folder}"
            file_path = Path(str(dst_filename) + ".zip")

            try:
                folder_path = config.datasets.get(folder)["target"]
            except ConfigKeyError:
                folder_path = config.datasets.get(folder)["source"]

            shutil.make_archive(str(dst_filename), "zip", Path(folder_path))

            # create a tag with hash value
            # compute hash sum and assign to new tag
            tags = {S3FileTags.hash_value.value: compute_file_hash(file_path)}
            # upload file to s3
            url = URL(remote_save_path).anchor
            upload_url = S3Handler(url=url, credentials=credentials).upload_file(
                src_path=file_path,
                dst_path=remote_save_path,
                tags=tags,
            )
            if upload_url is None:
                raise ValueError("Could not upload the dataset file")

    return upload_url


if __name__ == "__main__":
    Fire(upload_dataset)
