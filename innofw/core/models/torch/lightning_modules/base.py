# standard libraries
from typing import Dict, Any

# third party libraries
import pytorch_lightning as pl


class BaseLightningModule(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        ref: https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_advanced.html
        """
        model_name = "something"
        path_to_model = "something"
        metadata = {
            model_name: {
                "_target_": path_to_model,
                "weights": "something",
            },
            "metadata": {
                "description": "something",
                "data": "path",
                "metrics": {"recall": 0.56},
            },
        }
        checkpoint["_target_"] = "something"

    # def on_load_checkpoint(self, checkpoint):
    #     my_cool_pickable_object = checkpoint["something_cool_i_want_to_save"]
