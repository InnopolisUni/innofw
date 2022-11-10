# standard libraries
from typing import Dict, Any

# third party libraries
import pytorch_lightning as pl


class BaseLightningModule(pl.LightningModule):
    """
        Class that defines an interface for lightning modules
    """
