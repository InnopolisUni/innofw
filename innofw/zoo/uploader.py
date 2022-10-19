# standard libraries
from pathlib import Path
from typing import Optional

# third-party libraries
from fire import Fire
from pydantic import validate_arguments, AnyUrl
from urlpath import URL

# local modules
from innofw.constants import (
    S3Credentials,
    S3FileTags,
)
from innofw.utils.checkpoint_utils import add_metadata2model
from innofw.schema.model import ModelConfig
from innofw.utils import get_abs_path
from innofw.utils.file_hash import compute_file_hash
from innofw.utils.s3_utils import S3Handler
from innofw.utils.s3_utils.credentials import get_s3_credentials
from innofw.utils.s3_utils.minio_interface import get_full_dst_url


# todo: fix kwargs


@validate_arguments
def upload_model(
    ckpt_path: Path,
    config_save_path: Path,
    remote_save_path: AnyUrl,
    target: str,
    data: str,
    description: str,
    name: str,
    metrics: dict,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    **kwargs,
) -> None:
    """Function to upload a model weights into s3(remote storage) and generate config file for the model

    Arguments:
        ckpt_path - path to the local file with model weights. Can be relative to the project folder path
        config_save_path - path where to store the config file of the model. Can be relative to the project folder path
        remote_save_path - url to the file or file folder
            Example:
             1. https://api.blackhole.ai.innopolis.university/pretrained/
             2. https://api.blackhole.ai.innopolis.university/pretrained/catboost_qm9.cbm
        target - target class or callable name
        data - path to the data or name of the dataset from s3. If path is specified it is recommended to use relative to the project folder path.
        name - name of the model
        metrics - dictionary with keys as metrics name and values as metrics scores.
            Example:
                {'f1_score': 0.5}

                in cli use:
                    --metrics '{"mse": 0.04}'
        access_key - access key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials
        secret_key - secret key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials

    Usage:

    >>> from innofw.zoo import upload_model

    >>> upload_model(
    ...     ckpt_path = "pretrained/best.pkl",
    ...     config_save_path = "config/models/result.yaml",
    ...     remote_save_path = "https://api.blackhole.ai.innopolis.university/pretrained/model.pickle",
    ...     target = "sklearn.linear_models.LinearRegression",
    ...     data = "some/path/to/data",
    ...     description = "some description",
    ...     name = "some name",
    ...     metrics = {"some metric": 0.04},
    ...     access_key = "some key",
    ...     secret_key = "some secret"
    ...     )

    in cli:
        python innofw/zoo/uploader.py --ckpt_path pretrained/best.pkl\
                                      --config_save_path config/models/result.yaml\
                                      --remote_save_path https://api.blackhole.ai.innopolis.university/pretrained/model.pickle\
                                      --access_key $access_key --secret_key $secret_key\
                                      --target sklearn.linear_model.LinearRegression\
                                      --data some/path/to/data\
                                      --description "some description"\
                                      --metrics '{"some metric": 0.04}'\
                                      --name something
    """
    if access_key is None or secret_key is None:
        credentials = get_s3_credentials()
    else:
        credentials = S3Credentials(ACCESS_KEY=access_key, SECRET_KEY=secret_key)

    if not ckpt_path.is_absolute():
        ckpt_path = get_abs_path(ckpt_path)

    url = URL(remote_save_path).anchor
    s3handler = S3Handler(url, credentials)

    exp_upload_url = get_full_dst_url(ckpt_path, remote_save_path)

    metadata = {
        "_target_": target,
        "weights": exp_upload_url,
        "data": data,
        "name": name,
        "description": description,
        "metrics": metrics,
    }

    # create a tag with hash value
    tags = {
        S3FileTags.hash_value.value: compute_file_hash(ckpt_path),
    }  # compute hash sum and assign to new tag

    for key, value in metadata.items():
        tags[key] = str(value).encode()

    # upload file to s3
    upload_url = s3handler.upload_file(
        src_path=ckpt_path,
        dst_path=remote_save_path,
        tags=tags,
    )

    assert upload_url == exp_upload_url

    add_metadata2model(ckpt_path, metadata)

    # config file creation with specified s3 weights path
    model_cfg = ModelConfig(
        **kwargs,
        _target_=target,
        name=name,
        description=description,
        ckpt_path=upload_url,
    )
    # save the config file
    config_save_path = get_abs_path(config_save_path)
    config_save_path.parent.mkdir(exist_ok=True, parents=True)
    model_cfg.save_as_yaml(config_save_path)


if __name__ == "__main__":
    Fire(upload_model)
