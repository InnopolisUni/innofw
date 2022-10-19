# # standard libraries
# from pathlib import Path
#
# # third-party libraries
# import pytest
#
# # local modules
# from innframework.constants import DefaultS3User
from innofw.zoo import upload_model

# from tests.utils import get_test_folder_path
#
#
# @pytest.mark.parametrize(
#     ["file_path", "credentials"],
#     [
#         [
#             get_test_folder_path() / "weights/clustering_credit_cards/best.pkl",
#             DefaultS3User,
#         ]
#     ],
# )
# def test_upload_model(
#     file_path,
#     tmp_path,
#     credentials,
# ):
#     config_save_path = tmp_path / "something.yaml"
#     remote_save_path = Path("test/folder")
#
#     assert not config_save_path.exists()
#
#     config_args = {
#         "name": "linear regression cloud",
#         "description": "something 12",
#         "_target_": "innofw.models.something",
#     }
#     # todo: find out how to test function with user input of credentials
#     # upload_model(
#     #     file_path, config_save_path, remote_save_path, credentials, **config_args
#     # )
#
#     # assert config_save_path.exists()
#     # todo: add info that there should be additional fields
#     # todo: check config_file_contents
#     # read the yaml file
#     # convert contents into a dataset class
