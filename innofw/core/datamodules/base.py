# standard libraries
import os
from abc import ABC, abstractmethod

# third party libraries
from urllib.parse import urlparse
import pathlib
from pathlib import Path

# local modules

# todo: use pathlib instead of os.listdir, os.path.join
from innofw.constants import Stages
from innofw.data_mart import upload_dataset

from innofw.data_mart.downloader import download_archive
from innofw.utils import get_default_data_save_dir, get_abs_path


class BaseDataModule(ABC):
    task = ["None"]
    framework = ["None"]  # todo: why ["None"]

    def __init__(
        self,
        train=None,
        test=None,
        infer=None,
        stage=Stages.train,
        *args,
        **kwargs,
    ):
        if stage != Stages.predict:
            self.train = self._get_data(train)
            self.test = self._get_data(test)  # todo: following code could be async
        else:
            self.train = None
            self.test = None
        self.infer = self._get_data(infer)
        self.stage = stage

    def setup(self, stage: Stages = None, *args, **kwargs):
        if stage == Stages.predict or self.stage == Stages.predict:
            self.setup_infer()
        else:
            self.setup_train_test_val()

    @abstractmethod
    def setup_train_test_val(self):
        ...

    @abstractmethod
    def setup_infer(self):
        ...

    def get_stage_dataloader(self, stage):
        if stage is Stages.train:
            return self.train_dataloader()
        elif stage is Stages.test:
            return self.test_dataloader()
        elif stage is Stages.predict:
            return self.predict_dataloader()
        else:
            raise ValueError("Wrong stage passed use on of following:", list(Stages))

    @abstractmethod
    def train_dataloader(self):
        ...

    @abstractmethod
    def test_dataloader(self):
        ...

    @abstractmethod
    def predict_dataloader(self):
        pass

    @abstractmethod
    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        pass

    def _get_data(self, path):
        if path is None:
            return None

        for name in [
            "source",
        ]:  # , "target"
            if name not in path:
                raise ValueError(f"{name} path is not specified")

        source = path["source"]

        if source.startswith("$"):
            source = os.getenv(source[1:])

        if source.startswith("rts"):
            return source

        # target = path["target"]

        # source is not a link then return it
        if not urlparse(source).netloc:
            return get_abs_path(source)

        if "target" not in path:
            target = get_default_data_save_dir()
        else:
            target = Path(path["target"])

            if not target.is_absolute():
                target = get_abs_path(target)

        # process a link
        return download_archive(file_url=source, dst_path=target)

    def upload_dataset2s3(
        self,
        folder_path,
        config_save_path,
        remote_save_path,
        task,
        framework,
        target,
        name,
        description,
        markup_info,
        date_time,
        access_key,
        secret_key,
        **kwargs,
    ):
        upload_dataset(
            folder_path,
            config_save_path,
            remote_save_path,
            task,
            framework,
            target,
            name,
            description,
            markup_info,
            date_time,
            access_key,
            secret_key,
            **kwargs,
        )
