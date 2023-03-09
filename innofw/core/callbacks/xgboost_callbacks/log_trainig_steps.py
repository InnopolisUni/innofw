import xgboost as xgb
from torch.utils.tensorboard import SummaryWriter


class XGBoostTrainingTensorBoardCallback(xgb.callback.TrainingCallback):
    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir
        self.train_writer = SummaryWriter(log_dir=self.log_dir)
        self.test_writer = SummaryWriter(log_dir=self.log_dir)

    def after_iteration(
        self,
        model,
        epoch: int,
        evals_log: xgb.callback.TrainingCallback.EvalsLog,
    ) -> bool:
        if not evals_log:
            return False

        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                if data == "train":
                    self.train_writer.add_scalar(metric_name, score, epoch)
                else:
                    self.test_writer.add_scalar(metric_name, score, epoch)

        return False
