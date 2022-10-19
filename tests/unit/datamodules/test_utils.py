#
from pathlib import Path

import pytest

# local modules
from innofw.utils.dm_utils.utils import find_file_by_ext


@pytest.mark.parametrize(["ext"], [[".csv"], [".txt"], [".png"], [".jpg"]])
def test_search(tmp_path: Path, ext: str):
    # case: folder with a file with target extension
    # create a file in tmp_path
    f1 = tmp_path / "files" / f"file1{ext}"
    f1.parent.mkdir()
    f1.touch()
    # search for files
    file = find_file_by_ext(f1.parent, ext)
    # assert found file is exactly the needed file
    assert file == f1

    # case: folder with another file with different extension
    # create a file in tmp_path
    f2 = tmp_path / "files" / "file2.something"
    f2.touch()
    # search for files
    file = find_file_by_ext(f1.parent, ext)
    # assert found file is exactly the needed file
    assert file == f1

    # case: multiple files with same extension
    # create a file in tmp_path
    f3 = tmp_path / "files" / f"file3{ext}"
    f3.touch()
    # search for files
    file = find_file_by_ext(f1.parent, ext)
    # assert found file is exactly the needed file
    assert file == f1 or file == f3
