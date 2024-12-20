import argparse
from unittest.mock import patch

from innofw.utils.data_utils.rtk.lungs_description_metrics import (
    callback,
    setup_parser,
)


def test_setup_parser():
    parser = argparse.ArgumentParser()
    setup_parser(parser)
    args = parser.parse_args(["-i", "input.txt", "-o", "output.txt"])
    assert args.input == "input.txt"
    assert args.output == "output.txt"


@patch("innofw.utils.data_utils.rtk.lungs_description_metrics.calculate_lungs_metrics")
def test_callback_success(mock_calc):
    arguments = argparse.Namespace(input="input.txt", output="output.txt")
    callback(arguments)
    mock_calc.assert_called_once_with("input.txt", "output.txt")
