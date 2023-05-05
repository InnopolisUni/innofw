import pickle
from unittest.mock import patch

import pytest
import torch

<<<<<<< HEAD
from innofw.constants import CheckpointFieldKeys

=======
>>>>>>> af17e1e9e4f98c9d5200d1547671ab5abba55855

def t_execute_w_credentials(func):
    def executor_w_credentials(*args, **kwargs):
        return func(*args, **kwargs)

    return executor_w_credentials


patch(
    "innofw.utils.executors.execute_w_creds.execute_w_credentials",
    t_execute_w_credentials,
).start()

from innofw.zoo.show_model_metadata import show_model_metadata

tags = {
    "Accuracy": 0.85,
    "Precision": 0.75,
    "Recall": 0.80,
    "F1 Score": 0.77,
    "ROC AUC": 0.90,
    "Confusion Matrix": [[500, 50], [100, 350]],
    "Mean Absolute Error": 0.10,
    "Mean Squared Error": 0.02,
    "R Squared": 0.60,
    "Explained Variance": 0.70,
}


<<<<<<< HEAD
=======
def t_load_metadata(*args):
    return tags


>>>>>>> af17e1e9e4f98c9d5200d1547671ab5abba55855
def t_get_object_tags(*args):
    return tags


<<<<<<< HEAD
@patch("minio.api.Minio.get_object_tags", side_effect=t_get_object_tags)
@pytest.mark.parametrize(
    ["ckpt_path"],
    [
        [
            "https://api.blackhole.ai.innopolis.university/pretrained/testing/lin_reg_house_prices.pickle",
        ]
    ],
)
def test_show_model_metadata(mock_get_object_tags, ckpt_path, tmp_path):
    content = {
        CheckpointFieldKeys.model: torch.nn.Module().state_dict(),
        CheckpointFieldKeys.metadata: tags,
    }

    # Test case 1: ckpt_path is url
    metadata = show_model_metadata(ckpt_path)

    mock_get_object_tags.assert_called_once()
=======
@patch(
    "innofw.utils.checkpoint_utils.base_checkpoint_handler.CheckpointHandler.load_metadata",
    t_load_metadata,
)
@patch("minio.api.Minio.get_object_tags", t_get_object_tags)
@pytest.mark.parametrize(
    ["ckpt_path", "tags"],
    [
        [
            "https://api.blackhole.ai.innopolis.university/pretrained/testing/lin_reg_house_prices.pickle",
            {
                "Accuracy": 0.85,
                "Precision": 0.75,
                "Recall": 0.80,
                "F1 Score": 0.77,
                "ROC AUC": 0.90,
                "Confusion Matrix": [[500, 50], [100, 350]],
                "Mean Absolute Error": 0.10,
                "Mean Squared Error": 0.02,
                "R Squared": 0.60,
                "Explained Variance": 0.70,
            },
        ]
    ],
)
def test_show_model_metadata(ckpt_path, tmp_path, tags):
    # Test case 1: ckpt_path is url
    metadata = show_model_metadata(ckpt_path)

    assert metadata == tags
>>>>>>> af17e1e9e4f98c9d5200d1547671ab5abba55855

    # Test case 2: ckpt_path is file path with .pkl/.pickle/.cmb extension
    ckpt_path = tmp_path / "model.pkl"
    with open(ckpt_path, "wb+") as f:
<<<<<<< HEAD
        pickle.dump(content, f)
    metadata = show_model_metadata(ckpt_path)
=======
        pickle.dump(torch.nn.Module(), f)
    metadata = show_model_metadata(ckpt_path)

>>>>>>> af17e1e9e4f98c9d5200d1547671ab5abba55855
    assert metadata == tags

    # Test case 3: ckpt_path is file path with .ckpt/.pt extension
    ckpt_path = tmp_path / "model.pt"
    with open(ckpt_path, "wb+") as f:
<<<<<<< HEAD
        torch.save(content, f)
    metadata = show_model_metadata(ckpt_path)
=======
        torch.save(torch.nn.Module().state_dict(), f)
    metadata = show_model_metadata(ckpt_path)

>>>>>>> af17e1e9e4f98c9d5200d1547671ab5abba55855
    assert metadata == tags
