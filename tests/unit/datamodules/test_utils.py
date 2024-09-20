#
from pathlib import Path

import pytest

from innofw.utils.dm_utils.utils import find_file_by_ext, find_folder_with_images, query_yes_no

# local modules


@pytest.mark.parametrize(["ext"], [[".csv"], [".txt"], [".png"], [".jpg"]])
def test_search_by_ext(tmp_path: Path, ext: str):
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

@pytest.mark.parametrize(["ext"], [[".jpg"], [".png"], [".dcm"]])
def test_search_images(tmp_path: Path, ext: str):
    # case: one correct dir
    images_dir = tmp_path / 'images'
    img_1 = images_dir / f'1{ext}'
    img_1.parent.mkdir()
    img_1.touch()
    # found = find_folder_with_images(tmp_path)
    # assert found == images_dir

    # case: two images dirs were found
    images_dir_2 = tmp_path / 'images2'
    img_2 = images_dir_2 / f'1{ext}'
    img_2.parent.mkdir()
    img_2.touch()
    assert find_folder_with_images(tmp_path) is None

    no_images_dir = tmp_path / 'empty'
    no_images_dir.mkdir()
    assert find_folder_with_images(no_images_dir) is None
