# third party libraries
from omegaconf import DictConfig

# local modules
from tests.utils import get_test_folder_path

lep_datamodule_cfg_w_target = DictConfig(
    {
        "task": ["image-detection"],
        "name": "lep",
        "description": "object detection dataset of lep",
        "markup_info": "Информация о разметке",
        "date_time": "01.07.2076",
        "_target_": "innofw.core.integrations.YOLOv5DataModule",
        "train": {
            "source": str(get_test_folder_path() / "data/images/detection/lep/train")
        },
        "test": {
            "source": str(get_test_folder_path() / "data/images/detection/lep/test")
        },
        "infer": {
            "source": str(get_test_folder_path() / "data/images/detection/lep/infer")
        },
        "num_workers": 8,
        "val_size": 0.2,
        "image_size": 128,
        "num_classes": 4,
        "batch_size": 2,
        "names": ["lep_1", "lep_2", "lep_3", "lep_4"],
    }
)

# remote_lep_datamodule_cfg_w_target = DictConfig(
#     {
#         "task": ["image-detection"],
#         "name": "lep",
#         "description": "object detection dataset of lep",
#         "markup_info": "Информация о разметке",
#         "date_time": "01.07.2076",
#         "_target_": "innofw.datamodules.lightning_datamodules.detection.YOLOv5DataModule",
#         "train": {
#             "source": str(
#                 get_test_folder_path() / "data/images/detection/lep/train"
#             )
#         },
#         "test": {
#             "source": str(
#                 get_test_folder_path() / "data/images/detection/lep/test"
#             )
#         },
#         "num_workers": 8,
#         "val_size": 0.2,
#         "image_size": 128,
#         "num_classes": 4,
#         "names": ["lep_1", "lep_2", "lep_3", "lep_4"],
#     }
# )

house_prices_datamodule_cfg_w_target = DictConfig(
    {
        "task": ["table-regression"],
        "name": "house prices",
        "description": "",
        "markup_info": "",
        "date_time": "01.07.2022",
        "_target_": "innofw.core.datamodules.pandas_datamodules.PandasDataModule",
        "train": {
            "source": str(
                get_test_folder_path()
                / "data/tabular/regression/house_prices/train/train.csv"
            )
        },
        "test": {
            "source": str(
                get_test_folder_path()
                / "data/tabular/regression/house_prices/test/test.csv"
            )
        },
        "target_col": "price",
    }
)

wheat_datamodule_cfg_w_target = DictConfig(
    {
        "task": ["image-detection"],
        "name": "wheat",
        "description": "object detection dataset of wheat",
        "markup_info": "Информация о разметке",
        "date_time": "01.07.2076",
        "_target_": "innofw.core.datamodules.lightning_datamodules.detection_coco.CocoLightningDataModule",
        "train": {
            "source": str(get_test_folder_path() / "data/images/detection/wheat")
        },
        "test": {"source": str(get_test_folder_path() / "data/images/detection/wheat")},
        "num_workers": 8,
    }
)

dicom_datamodule_cfg_w_target = DictConfig(
    {
        "task": ["image-detection"],
        "name": "stroke",
        "description": "object detection dataset of wheat",
        "markup_info": "Информация о разметке",
        "date_time": "01.07.2076",
        "_target_": "innofw.core.datamodules.lightning_datamodules.detection_coco.DicomCocoLightningDataModule",
        "train": {
            "source": str(get_test_folder_path() / "data/images/detection/stroke/test/")
        },
        "test": {
            "source": str(get_test_folder_path() / "data/images/detection/stroke/test/")
        },
        "num_workers": 1,
    }
)

arable_segmentation_cfg_w_target = DictConfig(
    {
        "task": ["image-segmentation"],
        "name": "arable",
        "description": "something",
        "markup_info": "something",
        "date_time": "04.08.2022",
        "_target_": "innofw.core.datamodules.lightning_datamodules.segmentation_hdf5_dm.HDF5LightningDataModule",
        "train": {
            "source": str(
                get_test_folder_path()
                / "data/images/segmentation/arable/train"
            )
        },
        "test": {
            "source": str(
                get_test_folder_path()
                / "data/images/segmentation/arable/test"
            )
        },
        "channels_num": 4,
    }
)

faces_datamodule_cfg_w_target = DictConfig(
    {
        "task": ["image-classification"],
        "name": "face-recognition",
        "description": "face-recognition",
        "markup_info": "Информация о разметке",
        "date_time": "01.07.2076",
        "_target_": "innofw.core.datamodules.lightning_datamodules.image_folder_dm.ImageLightningDataModule",
        "train": {
            "source": str(
                get_test_folder_path()
                / "data/images/classification/office-character-classification"
            )
        },
        "test": {
            "source": str(
                get_test_folder_path()
                / "data/images/classification/office-character-classification"
            )
        },
        "num_workers": 8,
        "val_size": 0.2,
        "batch_size": 2,
    }
)

qm9_datamodule_cfg_w_target = DictConfig(
    {
        "task": ["qsar-regression"],
        "name": "qm9",
        "description": "https://paperswithcode.com/dataset/qm9",
        "markup_info": "Информация о разметке",
        "date_time": "18.06.2019",
        "_target_": "innofw.core.datamodules.pandas_datamodules.QsarDataModule",
        "train": {
            "source": str(
                get_test_folder_path() / "data/tabular/molecular/smiles/qm9/train"
            )
        },
        "test": {
            "source": str(
                get_test_folder_path() / "data/tabular/molecular/smiles/qm9/train"
            )
        },
        "smiles_col": "smiles",
        "target_col": "gap",
    }
)
