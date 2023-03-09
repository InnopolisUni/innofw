#
import logging
from pprint import pprint
from typing import Optional
from typing import Union

from fire import Fire
from pydantic import AnyUrl
from pydantic import FilePath
from pydantic import validate_arguments
from urlpath import URL

from innofw.constants import S3Credentials
from innofw.utils import get_abs_path
from innofw.utils.checkpoint_utils import load_metadata
from innofw.utils.executors.execute_w_creds import execute_w_credentials
from innofw.utils.s3_utils import S3Handler
from innofw.utils.s3_utils.minio_interface import get_bucket_name
from innofw.utils.s3_utils.minio_interface import get_object_path

#
#

logging.getLogger().setLevel(logging.INFO)


@execute_w_credentials
def load_metadata_from_remote(ckpt_path, credentials) -> dict:
    url = URL(ckpt_path).anchor
    s3handler = S3Handler(url, credentials=credentials)
    bucket_name = get_bucket_name(ckpt_path)
    obj_path = get_object_path(ckpt_path)
    tags = s3handler.client.get_object_tags(bucket_name, obj_path)
    return tags


@validate_arguments
def show_model_metadata(
    file_path: Union[FilePath, AnyUrl],
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> None:
    """
    Arguments:
        file_path - path to remote or local checkpoint
            file_path is FilePath
                no creds needed
                just prints out the metadata

            file_path is AnyUrl
                tries without creds
                else tries to search for creds in env
                else asks for creds in cli

        access_key - access key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials
        secret_key - secret key to the remote storage server. If not specified .env will be searched, ~/.aws-credentials

    Usage:

    ckpt_path in following examples can be replaced with relative or absolute path to the local checkpoint files

    >>> from innofw.zoo import show_model_metadata

    >>> show_model_metadata(
    ...     file_path="https://api.blackhole.ai.innopolis.university/pretrained/model.pickle",
    ...     access_key="some key",
    ...     secret_key="some secret"
    ...     )


    in cli:
        python innofw/zoo/show_metadata.py\
                    --file_path https://api.blackhole.ai.innopolis.university/pretrained/model.pickle\
                    --access_key $access_key\
                    --secret_key $secret_key

    """
    url = URL(file_path)
    # process links
    if url.scheme != "" and url.netloc != "":
        metadata = load_metadata_from_remote(
            ckpt_path=file_path,
            credentials=S3Credentials(
                ACCESS_KEY=access_key, SECRET_KEY=secret_key
            ),
        )
    else:
        # process paths
        file_path = get_abs_path(file_path)
        metadata = load_metadata(file_path)

    pprint(metadata)
    # logging.info(metadata)


if __name__ == "__main__":
    Fire(show_model_metadata)
