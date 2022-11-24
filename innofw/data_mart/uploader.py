# standard libraries
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, List, Union

# third-party libraries
from fire import Fire
from urlpath import URL
from pydantic.types import DirectoryPath
from pydantic import AnyUrl, validate_arguments

# local modules
from innofw.schema.dataset import DatasetConfig
from innofw.constants import (
    S3Credentials,
    S3FileTags,
)
from innofw.utils import get_abs_path
from innofw.utils.file_hash import compute_file_hash
from innofw.utils.s3_utils.credentials import get_s3_credentials
from innofw.utils.s3_utils import S3Handler


@validate_arguments
def upload_dataset(
        folder_path: DirectoryPath,
        config_save_path: Path,
        remote_save_path: AnyUrl,
        task: Union[str, List[str]],
        framework: Union[str, List[str]],
        target: str,
        name: str,
        description: str,
        markup_info: str,
        date_time: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        **kwargs,
):
    """Function to upload a dataset into s3(remote storage) and generate config file for the dataset

    Arguments:
        folder_path - path to the local folder with `train`, `test` and `infer` inner folders.
            Can be relative to the project folder path
        config_save_path - path where to store the config file of the dataset.
            Can be relative to the project folder path
        remote_save_path - url to the dataset save location.
            New filename can be specified for the archive file
            Example:
             1. https://api.blackhole.ai.innopolis.university/public-datasets/
             2. https://api.blackhole.ai.innopolis.university/public-datasets/credit_cards.zip

        task - task type which can be solved by this dataset
        framework - frmaework type used with this datamodule
        target - target datamodule class
        name - name of the dataset
        description - description of the dataset
        markup_info - info for markup
        date_time - when data was created

        access_key - access key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials
        secret_key - secret key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials

    Usage:

    >>> from innofw.zoo import upload_model

    >>> upload_model(
    ...     folder_path = "data/industry_data",
    ...     config_save_path = "config/datasets/industry_data.yaml",
    ...     remote_save_path = "https://api.blackhole.ai.innopolis.university/public-datasets/industry_data.zip",
    ...     task="text-classification",
    ...     framework="sklearn",
    ...     target = "innofw.core.datamodules.lightning_datamodules.detection_coco.CocoLightningDataModule",
    ...     name = "some name",
    ...     description = "some description",
    ...     markup_info =  "something",
    ...     date_time = "01.01.22",
    ...     access_key = "some key",
    ...     secret_key = "some secret"
    ...     )

    in cli:
        python innofw/data_mart/uploader.py --folder_path data/industry_data\
                                            --config_save_path config/temp/dataset.yaml\
                                            --remote_save_path  https://api.blackhole.ai.innopolis.university/public-datasets/test_dataset/\
                                            --task image-regression\
                                            --framework torch\
                                            --target innofw.core.datamodules.lightning_datamodules.detection_coco.CocoLightningDataModule\
                                            --name "some dataset"\
                                            --description "something"\
                                            --markup_info "something"\
                                            --date_time "something"
    """
    if access_key is None or secret_key is None:
        credentials = get_s3_credentials()
    else:
        credentials = S3Credentials(ACCESS_KEY=access_key, SECRET_KEY=secret_key)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        remote_paths = dict()

        for folder in ["train", "test"]:
            dst_filename = tmpdir / f"{folder}"
            file_path = Path(str(dst_filename) + ".zip")

            shutil.make_archive(str(dst_filename), "zip", folder_path / folder)

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

            remote_paths[folder] = {"source": upload_url}

    # config file creation with specified s3 weights path
    dataset_cfg = DatasetConfig(
        **kwargs,
        **remote_paths,
        _target_=target,
        task=[task],
        name=name,
        description=description,
        markup_info=markup_info,
        date_time=date_time,
        framework=framework,
    )
    # save the config file
    config_save_path = get_abs_path(config_save_path)
    dataset_cfg.save_as_yaml(config_save_path)

    return upload_url


if __name__ == "__main__":
    Fire(upload_dataset)
