import configparser
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import ValidationError

from innofw.constants import S3Credentials

#
#


def get_s3_credentials() -> Optional[S3Credentials]:
    access_key = None
    secret_key = None
    # try to use aws config file
    try:
        # aws config
        possible_aws_conf_path = Path("~/.aws/credentials").expanduser()
        config = configparser.RawConfigParser()
        config.read(possible_aws_conf_path)

        access_key = config["aws_access_key_id"]
        secret_key = config["aws_secret_key_id"]
    except Exception as e:
        pass

    # try to use env secrets
    try:
        load_dotenv()
        access_key = os.environ.get("access_key")
        secret_key = os.environ.get("secret_key")
    except Exception as e:
        pass

    try:
        # ask user
        if access_key is None:
            access_key: str = input("Provide S3 Access Key:")
        if secret_key is None:
            secret_key: str = input("Provide S3 Secret Key:")
    except:
        pass

    try:
        return S3Credentials(ACCESS_KEY=access_key, SECRET_KEY=secret_key)
    except ValidationError as e:
        print(e)

    return
