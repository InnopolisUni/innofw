# standard libraries
import importlib
import inspect

# third-party libraries
from torch.utils.tensorboard import SummaryWriter

# local modules
from .base import BaseModelAdapter
from innofw.core.callbacks.xgboost_callbacks.log_trainig_steps import (
    XGBoostTrainingTensorBoardCallback,
)
from innofw.utils.checkpoint_utils.pickle_checkpont_handler import (
    PickleCheckpointHandler,
)
from innofw.core.models import register_models_adapter


@register_models_adapter(name="xgboost_adapter")
class XGBoostAdapter(BaseModelAdapter):
    """
    Adapter for working with XGBoost models
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
    def _test(self, data):
        pass

    @staticmethod
    def is_suitable_model(model) -> bool:
        return inspect.getmodule(model).__package__.split(".")[0] == "xgboost"

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
            callable_metrics.append({"func": getattr(mod, func_name), "args": args})
        return callable_metrics

    def _predict(self, datamodule):
        data = datamodule.predict_dataloader()["x"]
        return self.model.predict(data)

    def forward(self, x, *args, **kwargs):
        return self.model.predict(x, *args, **kwargs)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def _train(self, datamodule, **kwargs):
        # datamodule.setup()
        data = datamodule.train_dataloader()
        x, y = data["x"], data["y"]
        # self.model.set_params(callbacks=[XGBoostTrainingTensorBoardCallback(log_dir=self.log_dir)])

        self.model.fit(
            x,
            y,
            eval_set=[(x, y), (x, y)],
        )
        self.test(datamodule)

    def test(self, datamodule):
        # datamodule.setup()
        data = datamodule.train_dataloader()
        x, y = data["x"], data["y"]
        results = {}
        y_pred = self.forward(x)
        for metric in self.metrics:
            score = metric["func"](y_pred, y, **metric["args"])
            results[metric["func"].__name__] = score
        self.log_results(results)
        return results

    def log_results(self, results):
        train_summary_writer = SummaryWriter(log_dir=self.log_dir)
        for metric, result in results.items():
            train_summary_writer.add_scalar(metric, result, 0)
            print(f"{metric}: {result}")
