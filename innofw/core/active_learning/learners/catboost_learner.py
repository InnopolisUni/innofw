import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from .base import BaseActiveLearner
from innofw import InnoModel


class LoggerPrint:
    @staticmethod
    def log(log_dict):
        print(log_dict)


class CatBoostActiveLearner(BaseActiveLearner):
    """
    A class for using CatBoost model in active learning.

    Attributes
    ----------
    model : InnoModel
        Wrapped CatBoost model
    datamodule : BaseDatamodule
        DataModule instance
    query_size : int
        Queue size for labeling request
    use_data_uncertainty: bool
        Use data uncertainty to select samples for labeling or not

    Methods
    -------
    eval_model(X, y):
        Evaluate trained model and return metrics.
    """

    def __init__(
        self,
        model: InnoModel,
        datamodule,
        epochs_num: int = 100,
        query_size: int = 1,
        use_data_uncertainty: bool = True,
    ):
        super().__init__(
            model, datamodule, epochs_num, query_size, logger=LoggerPrint()
        )
        self.use_data_uncertainty = use_data_uncertainty

    def eval_model(self, X, y):
        # ensemble_mae()
        preds = self._virtual_ensemble_prediction(X)
        y_preds = preds[0]
        mae = mean_absolute_error(y, y_preds)
        mse = mean_squared_error(y, y_preds)
        metrics = {
            "mae": f"{mae:2e}",
            "mse": f"{mse:2e}",
            "median model uncertainty": f"{np.median(preds[1]):2e}",
        }
        if len(preds) == 3 and self.use_data_uncertainty:
            metrics.update(
                {"median data uncertainty": f"{np.median(preds[2]):2e}"}
            )
        return metrics

    def predict_model(self, X):
        # ensemble_prediction()
        preds = self._virtual_ensemble_prediction(X)

        return preds

    def obtain_most_uncertain(self, preds):
        # most_uncertain_indices
        knowledge_uncertainty = preds[1]
        if len(preds) == 3 and self.use_data_uncertainty:
            data_uncertainty = preds[2]
            knowledge_uncertainty_normed = (
                knowledge_uncertainty / knowledge_uncertainty.max()
            )
            data_uncertainty_normed = data_uncertainty / data_uncertainty.max()

            uncertainty = (
                knowledge_uncertainty_normed - data_uncertainty_normed
            )
        else:
            uncertainty = knowledge_uncertainty

        kth_idx_most_uncertain = np.argpartition(
            uncertainty, -self.query_size
        )[-self.query_size :]

        return kth_idx_most_uncertain

    def _virtual_ensemble_prediction(
        self, X, virtual_ensembles_count: int = 10
    ):
        preds = self.model.model.virtual_ensembles_predict(
            X,
            prediction_type="TotalUncertainty",
            virtual_ensembles_count=virtual_ensembles_count,
        )

        out_preds = []
        # mean values predicted by a virtual ensemble
        mean_preds = preds[:, 0]
        out_preds.append(mean_preds)

        # knowledge uncertainty predicted by a virtual ensemble
        knowledge_uncertainty = preds[:, 1]
        out_preds.append(knowledge_uncertainty)

        if preds.shape[1] == 3 and self.use_data_uncertainty:
            # average estimated data uncertainty
            data_uncertainty = preds[:, 2]
            out_preds.append(data_uncertainty)

        return out_preds
