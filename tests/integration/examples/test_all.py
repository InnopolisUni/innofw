from innofw.utils import get_project_root
import logging
import subprocess
import pytest
from pathlib import Path
from time import sleep

examples_path = (get_project_root() / "examples/working").rglob("*.sh")  # working
examples_path = [[item] for item in examples_path]


@pytest.mark.skip(reason="Test is not applicable for all shell scripts")
@pytest.mark.parametrize(["command"], examples_path)
def test_shell_command(command, monkeypatch):
    logging.info(f"{Path(command.parent.name, command.name)}")
    inputs = iter(["y\n"] * 1000)
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    process = subprocess.Popen(
        ["sh", command], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    process.communicate(b"y\ny\ny\ny\n")
    # process.communicate(b'y')
    # process = subprocess.run(
    #     ["sh", command],
    #     # stdout=subprocess.PIPE,
    #     capture_output=True,
    #     universal_newlines=True,
    # )
    assert process.returncode == 0, f"Error message: {process.stdout}"
