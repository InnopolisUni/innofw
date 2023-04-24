from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score

from innofw.constants import Frameworks
from innofw.core.metrics.custom_metrics import BaseMetric
from innofw.core.metrics.custom_metrics import register_custom_metrics


@register_custom_metrics(
    task="classification",
    framework=[Frameworks.sklearn, Frameworks.xgboost],
    name="f1 score",
    description="something",
)
class F1Score(BaseMetric):
    """
    Custom f1-score
    ...

    Attributes
    ----------
    self.metric : metric
        metric function to perform
    Methods
    -------

    """

    def __init__(self):
        self.metric = f1_score
        super().__init__()


@register_custom_metrics(
    task="classification",
    framework=[Frameworks.sklearn, Frameworks.xgboost],
    name="accuracy",
    description="accuracy score",
)
class Accuracy(BaseMetric):
    """
    Custom Accuracy score
    ...

    Attributes
    ----------
    self.metric : metric
        metric function to perform
    Methods
    -------
    score(pred, labels):
        custom accuracy implementation
    """

    def __init__(self):
        self.metric = self.score
        super().__init__()

    def score(self, pred, labels):
        num_of_predictions = len(labels)
        correct = sum(
            [1 for i in range(num_of_predictions) if pred[i] == labels[i]]
        )
        return correct / float(num_of_predictions) * 100.0


@register_custom_metrics(
    task="classification",
    framework=[Frameworks.sklearn, Frameworks.xgboost],
    name="precision",
    description="Precision",
)
class Precision(BaseMetric):
    """
    Custom Precision score
    ...

    Attributes
    ----------
    self.metric : metric
        metric function to perform
    Methods
    -------

    """

    def __init__(self):
        self.metric = precision_score
        super().__init__()


@register_custom_metrics(
    task="classification",
    framework=[Frameworks.sklearn, Frameworks.xgboost],
    name="recall",
    description="Recall",
)
class Recall(BaseMetric):
    """
    Custom Recall score
    ...

    Attributes
    ----------
    self.metric : metric
        metric function to perform
    Methods
    -------
    score(pred, labels):
        custom accuracy implementation
    """

    def __init__(self):
        self.metric = self.score
        super().__init__()

    def score(self, pred, labels):
        tp = self.true_positive(labels, pred)
        fn = self.false_negative(labels, pred)
        return tp / (tp + fn)

    def true_positive(self, labels, pred):
        return sum(
            [1 for gt, pred in zip(labels, pred) if gt == 1 and pred == 1]
        )

    def false_negative(self, labels, pred):
        return sum(
            [1 for gt, pred in zip(labels, pred) if gt == 1 and pred == 0]
        )


@register_custom_metrics(
    task="regression",
    framework=[Frameworks.sklearn, Frameworks.xgboost],
    name="r2_score",
    description="R2_score",
)
class R2(BaseMetric):
    """
    Custom R2 score
    ...

    Attributes
    ----------
    self.metric : metric
        metric function to perform
    Methods
    -------

    """

    def __init__(self):
        self.metric = r2_score
        super().__init__()


@register_custom_metrics(
    task="regression",
    framework=[Frameworks.sklearn, Frameworks.xgboost],
    name="IOU",
    description="intersection over union",
)
class IOU(BaseMetric):
    """
    Custom intersection over union score
    ...

    Attributes
    ----------
    self.metric : metric
        metric function to perform
    Methods
    -------

    """

    def __init__(self):
        self.metric = jaccard_score
        super().__init__()
