from typing import Tuple
from unittest.mock import patch
import json
import os

from PIL import Image
import numpy as np
import pytest

from innofw.core.integrations.florence.florence_datamodule import (
    FlorenceImageDataset,
    FlorenceJSONLDataset,
    FlorenceImageDataModuleAdapter,
    FlorenceJSONLDataModuleAdapter,
)
from innofw.core.integrations.florence.florence_adapter import (
    FlorenceModel,
    FlorenceModelAdapter,
)
from innofw.core.integrations.florence.utils import init_tokenizer


DEFAULT_OUTPUT = [0, 0, 6940, 118, 4399, 571, 7776, 50634, 50502, 51142, 50820, 2, 1]


@pytest.fixture(scope="session")
def fake_gray_images(tmp_path_factory):
    """
    Fixture to generate three gray images in JPG format.

    Parameters:
    -----------
    tmp_path_factory : pathlib.Path
        Temporary directory provided by pytest for creating temporary files.
    """
    number_of_fake_images = 1
    size = (1000, 1000)
    test_pathology = "cardiomegaly"
    dicts = []

    temp_dir = tmp_path_factory.mktemp("data")
    os.makedirs(os.path.join(temp_dir, "images"))
    for i in range(number_of_fake_images):
        img = Image.new("L", size, color=128)  # 'L' mode for grayscale
        file_name = f"gray_image_{i + 1}.jpg"
        img_path = temp_dir / "images" / file_name
        img.save(img_path, "JPEG")
        dicts += [
            {
                "image": file_name,
                "prefix": "CAPTION_TO_PHRASE_GROUNDING " + test_pathology,
            }
        ]

    jsonl_file = temp_dir / "annotations.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for item in dicts:
            json.dump(item, f)
            f.write("\n")
    return temp_dir


@pytest.mark.parametrize("DS", [FlorenceImageDataset, FlorenceJSONLDataset])
def test_datasets(fake_gray_images, DS):
    ds = DS(fake_gray_images)
    for sample in ds:
        assert sample["image"]
        break


@pytest.mark.parametrize(
    "DM", [FlorenceImageDataModuleAdapter, FlorenceJSONLDataModuleAdapter]
)
def test_datamodules(fake_gray_images, DM):
    DEFAULT_DATAMODULE_CONFIG = {
        "size_to": 768,
        "infer": {"source": str(fake_gray_images)},
        "test": None,
        "train": None,
    }
    dm = DM(**DEFAULT_DATAMODULE_CONFIG)
    ds = dm.predict_dataloader()
    for sample in ds:
        assert "tensor" in sample
        break


def test_tokenizer_init():
    tokenizer_path = "innofw/core/integrations/florence/tokenizer"
    tokenizer = init_tokenizer(tokenizer_path)
    assert tokenizer


@pytest.fixture(scope="session")
def FlorenceModel_fixture():
    config = {
        "tokenizer_save_path": "innofw/core/integrations/florence/tokenizer/",
        "device": "cpu",
        "task_prompt": "<CAPTION_TO_PHRASE_GROUNDING>",
        "text_input": "cardiomegaly",
    }
    model = FlorenceModel(**config)
    return model


def test_FlorenceModel_initialize(FlorenceModel_fixture):
    assert FlorenceModel_fixture


@pytest.fixture(scope="session")
def FlorenceAdapter_fixt(
    tmp_path_factory, FlorenceModel_fixture
) -> Tuple[FlorenceModelAdapter, str]:
    log_dir = tmp_path_factory.mktemp("log_dir")
    trainer_cfg = {
        "accelerator": "cpu",
        "log_every_n_steps": 1,
        "max_epochs": None,
        "gpus": None,
        "devices": None,
    }
    model = FlorenceModel_fixture
    adapter = FlorenceModelAdapter(
        trainer_cfg=trainer_cfg, model=model, log_dir=log_dir
    )
    return adapter, log_dir


def test_FlorenceModelApadter_initialize(FlorenceAdapter_fixt):
    adapter, logs = FlorenceAdapter_fixt
    assert adapter


def test_infer_one_image(FlorenceAdapter_fixt, fake_gray_images):
    adapter, logs = FlorenceAdapter_fixt
    DEFAULT_DATAMODULE_CONFIG = {
        "size_to": 768,
        "infer": {"source": str(fake_gray_images)},
        "test": None,
        "train": None,
    }
    dm = FlorenceImageDataModuleAdapter(**DEFAULT_DATAMODULE_CONFIG)
    with patch(
        "innofw.core.integrations.florence.florence_adapter.FlorenceModel.get_generated_tokens"
    ) as Mocked:
        Mocked.return_value = DEFAULT_OUTPUT
        adapter.predict(dm)
    saved_files = os.listdir(logs)
    assert np.any(np.char.endswith(saved_files, "jpg"))
