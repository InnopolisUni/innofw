"""
This package is part of our framework's CORE, which is meant to give flexible support for callbacks out of different
libraries and frameworks via common abstract wrapper, currently it has support for:
- pytorch lightning
- sklearn
- xgboost

Callbacks are functions that can be called in the process of training as a response to some event.
"""

from .lightning_callbacks import log_segmentation_metrics, log_segmentation_predictions
