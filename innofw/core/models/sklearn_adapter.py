# standard libraries
import importlib
import inspect

import numpy as np
import sklearn
from torch.utils.tensorboard import SummaryWriter

from .base import BaseModelAdapter
from innofw.core.models import register_models_adapter
from innofw.utils.checkpoint_utils.pickle_checkpont_handler import (
    PickleCheckpointHandler,
)

# third party libraries/frameworks
# local modules


@register_models_adapter(name="sklearn_adapter")
class SklearnAdapter(BaseModelAdapter):
    """
    Adapter for working with Sklearn models
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

    @staticmethod
    def is_suitable_model(model) -> bool:
        return inspect.getmodule(model).__package__.split(".")[0] == "sklearn"

    def __init__(self, model, log_dir, callbacks=None, *args, **kwargs):
        super().__init__(model, log_dir, PickleCheckpointHandler())
        self.metrics = []
        if callbacks:
            self.metrics = self.prepare_metrics(callbacks)

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

    def forward(self, data):
        return self.model.predict(data)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def _predict(self, datamodule, ckpt_path=None, **kwargs):
        data = datamodule.predict_dataloader()
        x = data["x"]
        return self.model.predict(X=x)

    def _train(self, datamodule, **kwargs):
        data = datamodule.train_dataloader()
        x, y = data["x"], data["y"]
        try:
            self.model.steps = list(  self.model.steps)
        except:
            pass
        self.model.fit(X=x, y=y)
        self.test(datamodule)

    def _test(self, datamodule, **kwargs):
        data = datamodule.test_dataloader()
        x, y = data["x"], data["y"]
        results = {}
        y_pred = self.forward(x)
        for metric in self.metrics:
            if isinstance(self.model, sklearn.base.ClusterMixin):
                score = metric["func"](x, np.array(y_pred), **metric["args"])
            else:
                score = metric["func"](y_pred, y, **metric["args"])

            # name = get_func_name(metric)
            results[metric["func"].__name__] = score
        self.log_results(results)
        return results

    def log_results(self, results):
        train_summary_writer = SummaryWriter(log_dir=self.log_dir)
        for metric, result in results.items():
            train_summary_writer.add_scalar(metric, result, 0)
            print(f"{metric}: {result}")
