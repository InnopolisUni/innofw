# standard libraries
from pathlib import Path
from typing import Optional

from fire import Fire
from pydantic import AnyUrl
from pydantic import validate_arguments
from urlpath import URL
import hydra

from innofw.constants import S3Credentials
from innofw.constants import S3FileTags
from innofw.schema.model import ModelConfig
from innofw.utils import get_abs_path
from innofw.utils.checkpoint_utils import add_metadata2model
from innofw.utils.file_hash import compute_file_hash
from innofw.utils.s3_utils import S3Handler
from innofw.utils.s3_utils.credentials import get_s3_credentials
from innofw.utils.s3_utils.minio_interface import get_full_dst_url

# third-party libraries
# local modules


@validate_arguments
def upload_model(
    experiment_config_path: str,
    ckpt_path: Path,
    remote_save_path: AnyUrl,
    metrics: dict,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    **kwargs,
) -> None:
    """Function to upload a model weights into s3(remote storage) and generate config file for the model

    Arguments:
        experiment_config_path - path to the experiment config (relatively to config/experiments folder)   
        ckpt_path - path to the local file with model weights. Can be relative to the project folder path
        remote_save_path - url to the file or file folder
            Example:
             1. https://api.blackhole.ai.innopolis.university/pretrained/
             2. https://api.blackhole.ai.innopolis.university/pretrained/catboost_qm9.cbm
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
    ...     experiment_config_path = "classification/DS_190423_8dca23dc_ucmerced.yaml"
    ...     ckpt_path = "pretrained/best.pkl",
    ...     remote_save_path = "https://api.blackhole.ai.innopolis.university/pretrained/model.pickle",
    ...     metrics = {"some metric": 0.04},
    ...     access_key = "some key",
    ...     secret_key = "some secret"
    ...     )

    in cli:
        python innofw/zoo/uploader.py --experiment_config_path classification/DS_190423_8dca23dc_ucmerced.yaml
                                      --ckpt_path pretrained/best.pkl\
                                      --remote_save_path https://api.blackhole.ai.innopolis.university/pretrained/model.pickle\
                                      --metrics '{"some metric": 0.04}'
    """
    if access_key is None or secret_key is None:
        credentials = get_s3_credentials()
    else:
        credentials = S3Credentials(
            ACCESS_KEY=access_key, SECRET_KEY=secret_key
        )

    if not ckpt_path.is_absolute():
        ckpt_path = get_abs_path(ckpt_path)

    with hydra.initialize(config_path="../../config", version_base="1.2"):
        config = hydra.compose(config_name="train.yaml", overrides=[f"experiments={experiment_config_path}"])

    url = URL(remote_save_path).anchor
    s3handler = S3Handler(url, credentials)

    exp_upload_url = get_full_dst_url(ckpt_path, remote_save_path)
    train_data = config.get("datasets").get("train").target
    data = train_data[:train_data.find("/train")]

    metadata = {
        "_target_": config.models._target_,
        "weights": exp_upload_url,
        "data": data,
        "name": config.models.name,
        "description": config.models.description,
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


if __name__ == "__main__":
    Fire(upload_model)
