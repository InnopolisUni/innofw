from pathlib import Path


def get_test_folder_path() -> Path:
    return Path(__file__).parent.resolve()


def get_test_data_folder_path() -> Path:
    return get_test_folder_path() / "data"


def get_test_weights_folder_path() -> Path:
    return get_test_folder_path() / "weights"


def get_project_config_folder_path() -> Path:
    return get_test_folder_path().parent / "config"


def get_test_config_folder_path() -> Path:
    return get_test_folder_path() / "fixtures/config"


def is_dir_empty(dir_path):
    """
    faster way of finding if a directory is empty or not

    source: https://stackoverflow.com/a/57969033
    """
    path = Path(dir_path)
    has_next = next(path.iterdir(), None)
    if has_next is None:
        return True
    return False
