# standard libraries
import importlib
import logging

from catboost import CatBoost
from catboost import Pool
from torch.utils.tensorboard import SummaryWriter

from .base import BaseModelAdapter
from innofw.constants import CheckpointFieldKeys
from innofw.core.models import register_models_adapter
from innofw.utils.checkpoint_utils import PickleCheckpointHandler

# third party libraries/frameworks
# local modules

# from src.wrappers.utils import get_func_name


@register_models_adapter(name="catboost_adapter")
class CatBoostAdapter(BaseModelAdapter):
    """
    Adapter for working with CatBoost models
    ...

    Attributes
    ----------
    metrics : list
        List of metrics

    Methods
    -------
    log_results(results):
        logs metrics
    forward(x):
        returns result of prediction

    """

    model: CatBoost

    def __init__(
        self, model: CatBoost, log_dir, callbacks=None, *args, **kwargs
    ):
        super().__init__(model, log_dir, PickleCheckpointHandler())
        self.metrics = []
        if callbacks:
            self.metrics = self.prepare_metrics(callbacks)

    @staticmethod
    def is_suitable_model(model) -> bool:
        return isinstance(model, CatBoost)

    def prepare_metrics(self, metrics):
        callable_metrics = []
        for m in metrics:
            mod_name, func_name = m["_target_"].rsplit(".", 1)
            args = dict(m)
            del args["_target_"]
            mod = importlib.import_module(mod_name)
            callable_metrics.append(
                {"func": getattr(mod, func_name), "args": args}
            )
        return callable_metrics

    def _train(self, datamodule):
        data = datamodule.train_dataloader()

        x, y = data["x"], data["y"]
        cat_features = x.select_dtypes(include=["object"]).columns.tolist()
        train_pool = Pool(x, y, cat_features=cat_features)
        self.model.fit(train_pool)
        self.test(datamodule)

    def _test(self, datamodule):
        # datamodule.setup()
        data = datamodule.test_dataloader()
        x, y = data["x"], data["y"]
        cat_features = x.select_dtypes(include=["object"]).columns.tolist()
        test_pool = Pool(x, y, cat_features=cat_features)
        results = {}
        y_pred = self.model.predict(test_pool)
        if y_pred.ndim == 2:
            y_pred = y_pred[:, 0]

        for metric in self.metrics:
            score = metric["func"](y_pred, y, **metric["args"])

            # name = get_func_name(metric)
            results[metric["func"].__name__] = score
        self.log_results(results)
        return results

    def _predict(self, datamodule):
        if isinstance(self.model, dict):
            self.model = self.model[CheckpointFieldKeys.model]
        data = datamodule.predict_dataloader()
        x = data["x"]
        cat_features = x.select_dtypes(include=["object"]).columns.tolist()
        predict_pool = Pool(x, cat_features=cat_features)
        return self.model.predict(predict_pool)

    def _predict_proba(self, x):
        return self.model.predict_proba(x)

    def log_results(self, results):
        train_summary_writer = SummaryWriter(log_dir=self.log_dir)
        for metric, result in results.items():
            train_summary_writer.add_scalar(metric, result, 0)
            logging.info(f"{metric}: {result}")

    def virtual_ensembles_predict(
        self, data, prediction_type, virtual_ensembles_count
    ):
        return self.model.virtual_ensembles_predict(
            data=data,
            prediction_type=prediction_type,
            virtual_ensembles_count=virtual_ensembles_count,
        )
