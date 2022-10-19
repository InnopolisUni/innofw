from abc import ABC, abstractmethod

from innofw.core.datamodules.base import BaseDataModule
from innofw import InnoModel

from ..datamodule import get_active_datamodule


class BaseActiveLearner(ABC):
    def __init__(
        self,
        model: InnoModel,
        datamodule: BaseDataModule,
        epochs_num: int = 100,
        query_size: int = 1,
        logger=None,
    ):
        self.model = model
        self.active_datamodule = get_active_datamodule(datamodule)
        self.epochs_num = epochs_num
        self.query_size = query_size
        self.logger = logger

    def run(self, *, ckpt_path):
        for _ in range(self.epochs_num):
            if len(self.active_datamodule.pool_idxs) == 0:
                return

            self.model.train(self.active_datamodule, ckpt_path=ckpt_path)
            test_data = self.active_datamodule.test_dataloader()
            eval_metrics = self.eval_model(test_data["x"], test_data["y"])
            pool_data = self.active_datamodule.pool_dataloader()
            preds = self.predict_model(pool_data["x"])

            # log here time and
            if self.logger is not None:
                self.logger.log(
                    {
                        "train_samples": len(self.active_datamodule.train_idxs),
                        "pool_samples": len(self.active_datamodule.pool_idxs),
                        **eval_metrics,
                    }
                )

            most_uncertain = self.obtain_most_uncertain(preds)

            self.active_datamodule.update_indices(most_uncertain)

    @abstractmethod
    def eval_model(self, X, y):
        pass

    @abstractmethod
    def predict_model(self, X):
        pass

    @abstractmethod
    def obtain_most_uncertain(self):
        pass
