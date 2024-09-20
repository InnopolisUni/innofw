# third party libraries
import hydra
import torch.nn
from omegaconf import DictConfig
from omegaconf import OmegaConf

from innofw.core.models.torch.lightning_modules import (
    AnomalyDetectionTimeSeriesLightningModule,
    AnomalyDetectionImagesLightningModule
)
from innofw.core.models.torch.lightning_modules import BiobertNERModel
from innofw.core.models.torch.lightning_modules import (
    ChemistryVAEForwardLightningModule,
)
from innofw.core.models.torch.lightning_modules import (
    ChemistryVAELightningModule,
)
from innofw.core.models.torch.lightning_modules import (
    ChemistryVAEReverseLightningModule,
)
from innofw.core.models.torch.lightning_modules import (
    ClassificationLightningModule,
)
from innofw.core.models.torch.lightning_modules import (
    OneShotLearningLightningModule,
)
from innofw.core.models.torch.lightning_modules.detection import (
    DetectionLightningModule,
)
from innofw.core.models.torch.lightning_modules.segmentation import (
    MulticlassSemanticSegmentationLightningModule,
)
from innofw.core.models.torch.lightning_modules.segmentation import (
    SemanticSegmentationLightningModule,
)
from innofw.utils import get_logger
from innofw.utils.find_model import find_suitable_model

# local modules


logger = get_logger(__file__)


def default_model_for_datamodule(task, datamodule):
    defaults = {
        "table-regression": {
            "innofw.core.datamodules.pandas_datamodules.RegressionPandasDataModule": find_suitable_model(
                "linear_regression"
            ),
        }
    }
    return DictConfig(
        {
            "_target_": defaults[task][datamodule],
        }
    )


def get_default(obj_name: str, framework: str, task: str):
    default_scheduler_cfg = OmegaConf.create(
        {
            "task": ["all"],
            "implementations": {
                "torch": {
                    "StepLR": {
                        "object": {
                            "_target_": "torch.optim.lr_scheduler.ConstantLR",
                            "factor": 1,
                            "total_iters": 100,
                        }
                    }
                }
            },
        }
    )

    defaults = {
        "torch": {
            "image-segmentation": {
                "lightning_module": SemanticSegmentationLightningModule,
                "losses": [("CrossEntropy", 1, torch.nn.CrossEntropyLoss())],
                "optimizers_cfg": OmegaConf.create(
                    {"_target_": "torch.optim.Adam", "lr": 3e-4}
                ),
                "callbacks": [],
                "trainer_cfg": OmegaConf.create(
                    {
                        "_target_": "pytorch_lightning.Trainer",
                        "max_epochs": 100,
                    }
                ),
            },
            "multiclass-image-segmentation": {
                "lightning_module": MulticlassSemanticSegmentationLightningModule,
                "losses": [("CrossEntropy", 1, torch.nn.CrossEntropyLoss())],
                "optimizers_cfg": OmegaConf.create(
                    {"_target_": "torch.optim.Adam", "lr": 3e-4}
                ),
                "callbacks": [],
                "trainer_cfg": OmegaConf.create(
                    {
                        "_target_": "pytorch_lightning.Trainer",
                        "max_epochs": 100,
                    }
                ),
            },
            "image-detection": {
                "lightning_module": DetectionLightningModule,
                "optimizers_cfg": OmegaConf.create(
                    {"_target_": "torch.optim.Adam", "lr": 3e-4}
                ),
                "callbacks": [],
                "trainer_cfg": OmegaConf.create(
                    {
                        "_target_": "pytorch_lightning.Trainer",
                        "max_epochs": 100,
                    }
                ),
            },
            "one-shot-learning": {
                "lightning_module": OneShotLearningLightningModule,
                "optimizers_cfg": OmegaConf.create(
                    {"_target_": "torch.optim.Adam", "lr": 3e-4}
                ),
                "callbacks": [],
                "trainer_cfg": OmegaConf.create(
                    {
                        "_target_": "pytorch_lightning.Trainer",
                        "max_epochs": 100,
                    }
                ),
            },
            "image-classification": {
                "lightning_module": ClassificationLightningModule,
                "losses": torch.nn.NLLLoss(),
                "trainer_cfg": OmegaConf.create(
                    {"_target_": "pytorch_lightning.Trainer", "max_epochs": 1}
                ),
                "optimizers_cfg": OmegaConf.create(
                    {"_target_": "torch.optim.Adam", "lr": 3e-2}
                ),
            },
            "anomaly-detection-timeseries": {
                "lightning_module": AnomalyDetectionTimeSeriesLightningModule,
                "trainer_cfg": OmegaConf.create(
                    {"_target_": "pytorch_lightning.Trainer", "max_epochs": 1}
                ),
                "optimizers_cfg": OmegaConf.create(
                    {"_target_": "torch.optim.Adam", "lr": 3e-2}
                ),
            },
            "anomaly-detection-images": {
                "lightning_module": AnomalyDetectionImagesLightningModule,
                "trainer_cfg": OmegaConf.create(
                    {"_target_": "pytorch_lightning.Trainer", "max_epochs": 1}
                ),
                "optimizers_cfg": OmegaConf.create(
                    {"_target_": "torch.optim.Adam", "lr": 3e-2}
                ),
            },
            "text-ner": {
                "lightning_module": BiobertNERModel,
                "trainer_cfg": OmegaConf.create(
                    {"_target_": "pytorch_lightning.Trainer", "max_epochs": 1}
                ),
                "optimizers_cfg": OmegaConf.create(
                    {"_target_": "torch.optim.AdamW", "lr": 3e-2}
                ),
            },
            "text-vae": {
                "lightning_module": ChemistryVAELightningModule,
                "trainer_cfg": OmegaConf.create(
                    {"_target_": "pytorch_lightning.Trainer", "max_epochs": 1}
                ),
                "optimizers_cfg": OmegaConf.create(
                    {"_target_": "torch.optim.AdamW", "lr": 1e-2}
                ),
            },
            "text-vae-forward": {
                "lightning_module": ChemistryVAEForwardLightningModule,
                "trainer_cfg": OmegaConf.create(
                    {"_target_": "pytorch_lightning.Trainer", "max_epochs": 1}
                ),
                "optimizers_cfg": OmegaConf.create(
                    {"_target_": "torch.optim.AdamW", "lr": 1e-2}
                ),
            },
            "text-vae-reverse": {
                "lightning_module": ChemistryVAEReverseLightningModule,
                "trainer_cfg": OmegaConf.create(
                    {"_target_": "pytorch_lightning.Trainer", "max_epochs": 1}
                ),
                "optimizers_cfg": OmegaConf.create(
                    {"_target_": "torch.optim.AdamW", "lr": 1e-2}
                ),
            },
        }
    }
    try:
        if framework == "torch" and obj_name == "schedulers_cfg":
            logger.warning("using default scheduler")
            obj = default_scheduler_cfg
        else:
            obj = defaults[framework][task][obj_name]

        if isinstance(obj, DictConfig):
            return lambda *args, **kwargs: hydra.utils.instantiate(obj, *args, **kwargs)
        else:
            return obj
    except Exception as e:
        return None
