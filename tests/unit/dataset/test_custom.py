from pathlib import Path
from innofw.core.datasets.image_infer import ImageFolderInferDataset


def test_image_folder_dataset():
    dataset = ImageFolderInferDataset(
        image_dir=str(Path('tests/data/images/detection/lep/images/test/images/test/frame_001575.PNG'))
    )
    assert len(dataset) > 0
    assert dataset[0] is not None and len(dataset[0]) > 0