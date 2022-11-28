import pytorch_lightning as pl
from innofw import InnoModel
from innofw.core.models.catboost_adapter import CatBoostAdapter

from .learners import CatBoostActiveLearner


class ActiveLearnTrainer:
    """
    A class to train with active learn strategy.

    Attributes
    ----------
    learner : ActiveLearner 
        ActiveLearner instance

    Methods
    -------
    run(ckpt_path):
        Run active learning .
    """
    def __init__(
        self, model: InnoModel, datamodule: pl.LightningDataModule, *args, **kwargs
    ):
        # get some parameters
        self.learner = get_active_learner(model, datamodule, *args, **kwargs)

    def run(self, _, ckpt_path=None):
        self.learner.run(ckpt_path=ckpt_path)

    def predict(self):
        pass


def get_active_learner(
    model, datamodule, *args, **kwargs
):
    if isinstance(model.model, CatBoostAdapter):
        return CatBoostActiveLearner(model, datamodule, *args, **kwargs)
    else:
        raise NotImplementedError()
