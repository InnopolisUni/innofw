# third party libraries
import pytest
from tests.utils import get_test_folder_path

# local modules
from innofw.utils.s3_utils import S3Handler
from innofw.utils.s3_utils.minio_interface import (
    get_bucket_name,
    get_object_path,
    get_full_dst_path,
    get_full_dst_url,
    MinioInterface,
)
from innofw.utils.s3_utils.credentials import get_s3_credentials
from innofw.constants import (
    DefaultS3User,
    DEFAULT_STORAGE_URL,
    S3Credentials,
    Status,
    BucketNames,
)
from urlpath import URL

import logging

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize(
    ["url", "exp_bucket_name"],
    [
        [
            "https://api.blackhole.ai.innopolis.university/public-datasets/stroke_detection_dicom/test.zip",
            "public-datasets",
        ],
        [
            "https://api.blackhole.ai.innopolis.university/public-datasets/qm9/train.zip",
            "public-datasets",
        ],
        [
            "https://api.blackhole.ai.innopolis.university/pretrained/biobert_ner/epoch=0-step=100.ckpt",
            "pretrained",
        ],
        ["pretrained/biobert_ner/epoch=0-step=100.ckpt", "pretrained"],
    ],
)
def test_get_bucket_name(url, exp_bucket_name):
    bucket_name = get_bucket_name(url)
    assert bucket_name == exp_bucket_name


@pytest.mark.parametrize(
    ["url", "exp_obj_path"],
    [
        [
            "https://api.blackhole.ai.innopolis.university/public-datasets/stroke_detection_dicom/test.zip",
            "/stroke_detection_dicom/test.zip",
        ],
        [
            "https://api.blackhole.ai.innopolis.university/public-datasets/qm9/train.zip",
            "/qm9/train.zip",
        ],
        [
            "https://api.blackhole.ai.innopolis.university/pretrained/biobert_ner/epoch=0-step=100.ckpt",
            "/biobert_ner/epoch=0-step=100.ckpt",
        ],
        [
            "pretrained/biobert_ner/epoch=0-step=100.ckpt",
            "/biobert_ner/epoch=0-step=100.ckpt",
        ],
    ],
)
def test_get_object_path(url, exp_obj_path):
    obj_path = get_object_path(url)

    assert obj_path == exp_obj_path


# case 1: src_path path to the file, dst_path path to the file in the remote url
# case 2: src_path path to the file, dst_path path to the dir in the remote url
# case 3: some folders in between
@pytest.mark.parametrize(
    ["dst_path", "exp_dst_path"],
    [
        [
            "https://api.blackhole.ai.innopolis.university/pretrained",
            "https://api.blackhole.ai.innopolis.university/pretrained/file.txt",
        ],
        [
            "https://api.blackhole.ai.innopolis.university/pretrained/new_name.txt",
            "https://api.blackhole.ai.innopolis.university/pretrained/new_name.txt",
        ],
        [
            "https://api.blackhole.ai.innopolis.university/pretrained/some/folder/new_name.txt",
            "https://api.blackhole.ai.innopolis.university/pretrained/some/folder/new_name.txt",
        ],
    ],
)
def test_get_full_dst_url(tmp_path, dst_path, exp_dst_path):
    src_path = tmp_path / "file.txt"
    src_path.touch()

    new_dst_path = get_full_dst_url(src_path, dst_path)
    assert new_dst_path == exp_dst_path


@pytest.mark.parametrize(
    ["src_path"],
    [["https://api.blackhole.ai.innopolis.university/pretrained/file.txt"]],
)
def test_get_full_dst_path(src_path, tmp_path):
    dst_path = tmp_path
    exp_dst_path = tmp_path / URL(src_path).name
    new_dst_path = get_full_dst_path(src_path, dst_path)
    assert new_dst_path == exp_dst_path

    dst_path = tmp_path / "new_file.txt"
    new_dst_path = get_full_dst_path(src_path, dst_path)
    assert new_dst_path == dst_path


def test_s3handler_has_hash_value():
    url = DEFAULT_STORAGE_URL
    credentials = DefaultS3User
    s3handler = S3Handler(url, credentials)

    src_path = "https://api.blackhole.ai.innopolis.university/public-datasets/stroke_detection_dicom/test.zip"
    assert not s3handler.has_hash_value(src_path)

    # src_path = "https://api.blackhole.ai.innopolis.university/public-datasets/tests/file_with_hash.zip"
    # assert s3handler.has_hash_value(src_path)


def test_s3handler_should_download_file(tmp_path):
    url = DEFAULT_STORAGE_URL
    credentials = DefaultS3User
    s3handler = S3Handler(url, credentials)

    src_path = "https://api.blackhole.ai.innopolis.university/public-datasets/stroke_detection_dicom/test.zip"
    dst_path = tmp_path / "test.zip"
    assert s3handler.should_download_file(src_path, dst_path)


def test_minio_interface(tmp_path):
    url = DEFAULT_STORAGE_URL
    credentials = DefaultS3User
    s3handler = S3Handler(url, credentials)

    assert s3handler.client is not None

    src_path = "https://api.blackhole.ai.innopolis.university/public-datasets/credit_cards/test.zip"
    dst_path = tmp_path

    assert len(list(tmp_path.iterdir())) == 0
    save_path = s3handler.unsafe_download_file(src_path, dst_path)
    assert len(list(tmp_path.iterdir())) == 1
    assert save_path.exists()
    assert save_path.name == "test.zip"
    save_path.unlink()

    dst_path = tmp_path / "new_test.zip"
    assert len(list(tmp_path.iterdir())) == 0
    save_path = s3handler.unsafe_download_file(src_path, dst_path)
    assert len(list(tmp_path.iterdir())) == 1
    assert save_path.name == "new_test.zip"
    save_path.unlink()

    # assert len(list(tmp_path.iterdir())) == 0
    # save_path = s3handler.download_file(src_path, dst_path)
    # assert len(list(tmp_path.iterdir())) == 1
    # assert save_path.exists()
    # assert save_path.name == 'test.zip'
    # save_path.unlink()
    #
    # dst_path = tmp_path / 'new_test.zip'
    # assert len(list(tmp_path.iterdir())) == 0
    # save_path = s3handler.download_file(src_path, dst_path)
    # assert len(list(tmp_path.iterdir())) == 1
    # assert save_path.name == 'new_test.zip'
    # save_path.unlink()


# s3handler.upload_file

# s3handler.matching_hashes
# s3handler.has_hash_value
#
# s3handler.list_objects

# s3handler = S3Handler("url", "access_key", "secret_key")
# status, result = s3handler.upload_file("file_path", "dst_path", "tags")
# status, result = s3handler.download_file("remote_file_path", "dst_path")
# status, result = s3handler.upload_folder("folder_path", "dst_path", "tags")
# status, result = s3handler.download_folder("remote_folder_path", "dst_path")
# status, result = s3handler.get_tags("remote_file_path")
# status, result = s3handler.get_metadata("remote_file_path")
#
# # status, result = s3handler.remove_file("remote_file_path")
# # status, result = s3handler.mv_file("src_remote_file_path", "dst_remote_file_path")

import logging

# @pytest.mark.parametrize(
#     ["url", "credentials"],
#     [[DEFAULT_STORAGE_URL, DefaultS3User], [DEFAULT_STORAGE_URL, None]],
# )
# def test_s3handler_get_client(url, credentials):
#     s3handler = S3Handler(url, credentials)
#     buckets = s3handler.list_objects(BucketNames.model_zoo.value)
#     assert len(list(buckets)) != 0
#     buckets = s3handler.list_objects(BucketNames.data_mart.value)
#     assert len(list(buckets)) != 0

# assert s3handler is not None


# negative test
# wrong s3 credentials
WrongS3User = S3Credentials(
    ACCESS_KEY="One",
    SECRET_KEY="Two",
)

WrongS3Url = "https://google.com"


@pytest.mark.parametrize(
    ["url", "credentials"],
    [
        [WrongS3Url, DefaultS3User],
        [WrongS3Url, None],
        [DEFAULT_STORAGE_URL, WrongS3User],
        [WrongS3Url, WrongS3User],
    ],
)
def test_s3handler_get_client_neg(url, credentials):
    with pytest.raises(Exception):
        client = S3Handler.get_client(url, credentials)
        buckets = client.list_objects(BucketNames.model_zoo.value)
        assert len(list(buckets)) != 0
        buckets = client.list_objects(BucketNames.data_mart.value)
        assert len(list(buckets)) != 0
        assert client is not None


#
#
# @pytest.mark.parametrize(
#     ["file_url", "credentials"],
#     [
#         [
#             "https://api.blackhole.ai.innopolis.university/public-datasets/credit_cards/test.zip",
#             DefaultS3User,
#         ],
#     ],
# )
# def test_s3_handler_download_file(file_url, credentials, tmp_path):
#     S3Handler.safe_download_file(file_url, tmp_path, credentials)
#     assert len(list(tmp_path.iterdir())) != 0
#
# # @pytest.mark.parametrize(
# #     [
# #         "file_path",
# #         "bucket_name",
# #         "remote_save_path",
# #         "credentials",
# #         "storage_url",
# #         "tags",
# #         "expected_status",
# #     ],
# #     [
# #         [
# #             get_test_folder_path() / "weights/clustering_credit_cards/best.pkl",
# #             "pretrained",
# #             "test/best.pkl",
# #             DefaultS3User,
# #             DEFAULT_STORAGE_URL,
# #             {"hash_value": "test9824uq"},
# #             Status.FAIL,
# #         ],
# #         # [S3Credentials(
# #         #         ACCESS_KEY="test_user", SECRET_KEY="test_user"
# #         #     )]
# #     ],
# # )
# # def test_s3handler_upload_file(
# #     file_path,
# #     bucket_name,
# #     remote_save_path,
# #     credentials,
# #     storage_url,
# #     tags,
# #     expected_status,
# # ):
# #     upload_status, maybe_path = S3Handler.upload_file(
# #         file_path, bucket_name, remote_save_path, credentials, storage_url, tags
# #     )
# #
# #     assert upload_status == expected_status
# #     assert maybe_path is None
# #
# #
# # #     # positive test when status.success
# # #     assert maybe_path == "some/path"
