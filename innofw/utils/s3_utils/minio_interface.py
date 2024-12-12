# standard libraries
import os
from pathlib import Path
from typing import Optional

from minio import Minio
from minio.commonconfig import Tags
from pydantic import AnyUrl
from pydantic import FilePath
from pydantic import validate_arguments
from tqdm import tqdm
from urlpath import URL
import urllib3
from datetime import timedelta

from innofw.constants import S3Credentials
from innofw.utils import get_abs_path
from innofw.utils.dm_utils.utils import query_yes_no
from innofw.utils.extra import execute_with_retries
from innofw.utils.file_hash import compute_file_hash

# third-party libraries
# local modules
# from .credentials import get_s3_credentials

KB = 1024
MB = 1024 * KB


def get_bucket_name(url: AnyUrl) -> str:
    """Function returns name of the bucket for absolute and relative links"""
    url = URL(url)
    bucket_name = url.parts[1] if url.parts[0] == url.anchor else url.parts[0]
    return bucket_name


def get_object_path(url: AnyUrl) -> str:
    url = URL(url)
    anchor_n_bucket = url.anchor + get_bucket_name(url)
    object_path = str(url)[len(anchor_n_bucket) :]
    return object_path


@validate_arguments
def get_full_dst_url(src_path: FilePath, dst_path: AnyUrl) -> AnyUrl:
    url = URL(dst_path)
    if url.suffix == "":
        return str(url / src_path.name)

    return str(url)


@validate_arguments
def get_full_dst_path(
    src_path: AnyUrl, dst_path: Path, create_if_needed: bool = True
) -> Path:
    # new dst_path
    if dst_path.is_dir():
        dst_path = dst_path / URL(src_path).name

    if create_if_needed:
        # if dst_path.is_dir():
        #     dst_path.mkdir(parents=True, exist_ok=True)
        # else:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

    return dst_path


class MinioInterface:
    """
    Class used for working with data from s3 storage
    Attributes
    ----------
    client:
        minio client

    Methods
    -------
    upload_file(src_path: FilePath, dst_path: AnyUrl, tags: Optional[dict] = None):
        Method to upload file into minio server
    """

    def __init__(self, url, credentials):
        self.client = self._get_client(url, credentials)

    @validate_arguments
    def _get_client(
        self, url: AnyUrl, credentials: Optional[S3Credentials] = None
    ):
        url = URL(url).netloc
        # credentials = get_s3_credentials() if credentials is None else credentials
        http_client = None

        if "INSTALL_IGNORE_SSL" in os.environ and bool(os.environ["INSTALL_IGNORE_SSL"]) == True:
            timeout = timedelta(minutes=5).seconds
            http_client = urllib3.PoolManager(
                timeout=urllib3.util.Timeout(connect=timeout, read=timeout),
                maxsize=10,
                cert_reqs='CERT_NONE',
                assert_hostname=False,
                retries=urllib3.Retry(
                    total=5,
                    backoff_factor=0.2,
                    status_forcelist=[500, 502, 503, 504]))

        if credentials is None:
            client = Minio(url, access_key=None, secret_key=None, http_client=http_client)
        else:
            client = Minio(
                url,
                access_key=credentials.ACCESS_KEY
                if credentials.ACCESS_KEY is None
                else credentials.ACCESS_KEY.get_secret_value(),
                secret_key=credentials.SECRET_KEY
                if credentials.SECRET_KEY is None
                else credentials.SECRET_KEY.get_secret_value(),
                http_client=http_client
            )
        return client

    # @execute_with_retries
    @validate_arguments
    def upload_file(
        self,
        src_path: FilePath,
        dst_path: AnyUrl,
        tags: Optional[dict] = None,
    ):
        """Method to upload file into minio server

        Arguments:
            client: minio client
            src_path: path to the local file
            dst_path: url where file should be located.
                Possible urls:
                    - relative from the s3 server address link
                    - absolute link, meaning full url

                If url does not contain filename then file name will be taken from the source file
            tags: dictionary with additional information

        Returns:
        """

        # convert dict to minio.Tags
        minio_tags = Tags(for_object=True)
        minio_tags.update(**tags)

        dst_path = get_full_dst_url(src_path, dst_path)
        bucket_name = get_bucket_name(dst_path)
        obj_path = get_object_path(dst_path)

        # upload the file
        self.client.fput_object(
            bucket_name,
            obj_path,
            src_path,
            tags=minio_tags,
        )
        return dst_path

    def unsafe_download_file(
        self, src_path: AnyUrl, dst_path: Path, chunk_size=1 * MB
    ) -> Path:
        """Function downloads the file from minio server"""
        bucket = get_bucket_name(src_path)
        object_path = get_object_path(src_path)
        dst_path = get_full_dst_path(src_path, dst_path, create_if_needed=True)

        file_size = self.client.stat_object(bucket, object_path).size
        downloaded = 0 * MB

        with open(dst_path, "wb") as file_data, tqdm(
            desc=f"{dst_path.name}",
            total=file_size,
            dynamic_ncols=True,
            leave=False,
            mininterval=1,
            unit="B",
            unit_scale=True,
            unit_divisor=KB,
        ) as pbar:
            while downloaded < file_size:
                length = (
                    chunk_size
                    if (file_size - downloaded) >= chunk_size
                    else file_size - downloaded
                )
                data = self.client.get_object(
                    bucket, object_path, offset=downloaded, length=length
                )
                newly_downloaded = 0
                for d in data:
                    newly_downloaded += file_data.write(d)
                downloaded += newly_downloaded
                pbar.update(newly_downloaded)

        if downloaded != file_size:
            dst_path.unlink(missing_ok=True)
            raise Exception(
                f"File error: size of '{dst_path}' {downloaded} bytes, expected {file_size} bytes"
            )

        return dst_path

    @execute_with_retries
    def download_file(
        self, src_path: AnyUrl, dst_path: Path, chunk_size=1 * MB
    ) -> Path:
        if not dst_path.is_absolute():
            dst_path = get_abs_path(dst_path)
        if not dst_path.is_file():
            dst_path.mkdir(exist_ok=True, parents=True)

        dst_path = get_full_dst_path(src_path, dst_path, create_if_needed=True)
        do_downloading: bool = self.should_download_file(src_path, dst_path)

        if do_downloading:
            return self.unsafe_download_file(src_path, dst_path, chunk_size)

        return dst_path

    def matching_hashes(self, src_path, dst_path):
        bucket = get_bucket_name(src_path)
        obj_path = get_object_path(src_path)
        target_hash_value = self.client.get_object_tags(bucket, obj_path)[
            Tags.hash_value.value
        ]
        # compute local file's hash value
        computed_hash_value = compute_file_hash(dst_path)

        return target_hash_value == computed_hash_value

    def has_hash_value(self, src_path) -> bool:
        bucket = get_bucket_name(src_path)
        obj_path = get_object_path(src_path)
        try:
            target_hash_value = self.client.get_object_tags(bucket, obj_path)[
                Tags.hash_value.value
            ]
            return target_hash_value is not None
        except:
            return False

    def should_download_file(self, src_path, dst_path) -> bool:
        if not dst_path.exists():
            # file does not exist; download it
            return True

        if not self.has_hash_value(src_path):
            return query_yes_no(
                "Unable to retrieve hash from remote, but folder with same name exists, overwrite?",
                "no",
            )

        if self.matching_hashes(src_path, dst_path):
            return True  # hash values match; skip downloading

        # hash values don't match; ask user whether system should overwrite it
        return not query_yes_no(
            "Hash values of local file and downloaded file do not match, overwrite?",
            "no",
        )

    def list_objects(self, bucket_name: str):
        return self.client.list_objects(bucket_name)

    # def upload_folder(self, src_path: DirectoryPath, dst_path: AnyUrl, tags):
    #     pass
    #
    # def download_folder(self, src_path: AnyUrl, dst_path: Path):
    #     pass
    #
    # def get_tags(self, src_path: AnyUrl):
    #     pass
    #
    # def get_metadata(self, src_path: AnyUrl):
    #     pass
