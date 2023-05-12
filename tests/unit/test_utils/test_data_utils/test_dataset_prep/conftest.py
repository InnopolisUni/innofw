import pytest


@pytest.fixture(scope="module")
def data_dir(temp_dir):
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    # create dummy image and label files
    for i in range(10):
        with open(data_dir / f"{i}.jpeg", "w") as f:
            f.write("image file")
        with open(data_dir / f"{i}.txt", "w") as f:
            f.write("label file")
    return data_dir


@pytest.fixture(scope="module")
def train_names_file(temp_dir):
    train_names_file = temp_dir / "train.txt"
    with open(train_names_file, "w") as f:
        for i in range(5):
            f.write(f"{i}\n")
    return train_names_file


@pytest.fixture(scope="module")
def test_names_file(temp_dir):
    test_names_file = temp_dir / "test.txt"
    with open(test_names_file, "w") as f:
        for i in range(5, 10):
            f.write(f"{i}\n")
    return test_names_file
