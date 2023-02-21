from pathlib import Path

from omegaconf import DictConfig

from tests.utils import get_test_data_folder_path


roads_data_root = (
    get_test_data_folder_path() / "images/segmentation/linear-roads-bin-seg-oftp"
)


roads_tiff_dataset_w_masks = {
    "_target_": "innofw.core.datasets.semantic_segmentation.tiff_dataset.SegmentationDataset",
    "images": list((roads_data_root / "images").iterdir()),
    "masks": list((roads_data_root / "masks").iterdir()),
    "transform": None,
    "channels": None,
    "with_caching": False,
}
