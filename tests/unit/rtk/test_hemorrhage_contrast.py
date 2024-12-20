from unittest.mock import patch
import argparse

from innofw.utils.data_utils.preprocessing.CT_hemorrhage_contrast_rtk import (
    callback,
    default_output_path,
    setup_parser,
)


def test_setup_parser():
    parser = argparse.ArgumentParser()
    setup_parser(parser)
    args = parser.parse_args(["-i", "input.txt"])
    assert args.input == "input.txt"
    assert args.output is None

    args = parser.parse_args(["-i", "input.txt", "-o", "output.txt"])
    assert args.input == "input.txt"
    assert args.output == "output.txt"


def test_default_output_path():
    with patch("innofw.utils.getters.get_log_dir", return_value="mock_log_dir"):
        output_path = default_output_path()
        assert output_path == "mock_log_dir"


@patch(
    "innofw.utils.data_utils.preprocessing.CT_hemorrhage_contrast_rtk.hemorrhage_contrast"
)
def test_callback_success(mock_contrast):
    arguments = argparse.Namespace(input="input.txt", output="output.txt")
    callback(arguments)
    mock_contrast.assert_called_once_with("input.txt", "output.txt")
