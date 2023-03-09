#
import inspect

from innofw.core.metrics import register_metrics_adapter
from innofw.core.metrics.base import BaseMetricAdapter

#


@register_metrics_adapter("torchmetrics_adapter")
class TorchMetricsAdapter(BaseMetricAdapter):
    """
    Class for working with pytorch metrics
    ...

    Attributes
    ----------

    Methods
    -------
    forward(x):
        function to perform metric

    is_suitable_input(metric):
        function checks metric function
    """

    def __init__(self, metric, *args, **kwargs):
        super().__init__(metric)

    @staticmethod
    def is_suitable_input(metric) -> bool:
        if (
            inspect.getmodule(metric).__package__.split(".")[0]
            == "torchmetrics"
        ):
            return True
        return False

    def forward(self, *args, **kwargs):
        return self.metric(*args, **kwargs)
