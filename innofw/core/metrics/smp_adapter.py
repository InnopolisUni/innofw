import inspect

#
from innofw.core.metrics import register_metrics_adapter
from innofw.core.metrics.base import BaseMetricAdapter


@register_metrics_adapter("segmentation_models_pytorch_adapter")
class SMPMetricsAdapter(BaseMetricAdapter):
    """
    Class for working with smp metrics
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
            == "segmentation_models_pytorch"
        ):
            return True
        return False

    def forward(self, *args, **kwargs):
        self.metric.forward(*args, **kwargs)
