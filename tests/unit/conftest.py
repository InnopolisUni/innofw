import pytest
import tempfile
from pathlib import Path


@pytest.fixture(scope="module")
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        yield tmp_dir
