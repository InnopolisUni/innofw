import pickle
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from torch.nn import ModuleList
import pytorch_lightning as pl
from transformers import BertForTokenClassification, PreTrainedTokenizerBase


class BiobertNERModel(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        losses,
        optimizer_cfg,
        scheduler_cfg,
        *args,
        **kwargs,
    ):
        super().__init__(*args, *kwargs)

        self.model: BertForTokenClassification = model["model"]
        self.tokenizer: PreTrainedTokenizerBase = model["tokenizer"]

        loss_modules = [loss[2] for loss in losses]
        self.losses = ModuleList(loss_modules)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        assert self.losses is not None

    def __calc_loss(self, preds, targets):
        loss = 0
        for loss_f in self.losses:
            loss += loss_f(preds, targets)
        return loss

    def configure_optimizers(self):
        """Function to set up optimizers and schedulers"""
        # get all trainable model parameters
        params = [x for x in self.model.parameters() if x.requires_grad]
        # instantiate models from configurations
        optim = self.optimizer_cfg(params=params)
        # optim = hydra.utils.instantiate(self.optimizer_cfg, params=params)

        # instantiate scheduler from configurations
        try:
            scheduler = self.scheduler_cfg(optim)
            # scheduler = hydra.utils.instantiate(self.scheduler_cfg, optim)
            # return optimizers and schedulers
            return [optim], [scheduler]
        except:
            return [optim]

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings

    def training_step(self, X, batch_idx):
        input_ids, labels = X["input_ids"], X["labels"]
        result = self.model.forward(input_ids)
        logits = result["logits"]
        preds = logits.argmax(-1)
        mask = input_ids != self.tokenizer.pad_token_id
        loss = self.__calc_loss(logits[mask], labels[mask])

        return {"loss": loss, "preds": preds}

    def validation_step(self, X, batch_idx):
        input_ids, labels = X["input_ids"], X["labels"]
        result = self.model.forward(input_ids)
        logits = result["logits"]
        preds = logits.argmax(-1)
        mask = input_ids != self.tokenizer.pad_token_id
        loss = self.__calc_loss(logits[mask], labels[mask])

        return {"loss": loss, "preds": preds}

    def test_step(self, X, batch_idx):
        input_ids, labels = X["input_ids"], X["labels"]
        result = self.model.forward(input_ids)
        logits = result["logits"]
        preds = logits.argmax(-1)
        mask = input_ids != self.tokenizer.pad_token_id
        loss = self.__calc_loss(logits[mask], labels[mask])

        return {"loss": loss, "preds": preds}

    def predict_step(self, X, batch_idx):
        input_ids = X["input_ids"]
        result = self.model.forward(input_ids)
        logits = result["logits"]
        preds = logits.argmax(-1)

        return preds


class BiobertNERModelWithBIO(BiobertNERModel):
    def _get_label_ids_by_matched_target(self, match_target_indexes, entities):
        label_ids = []
        last_ind = None
        for match_target_index in match_target_indexes:
            if match_target_index is None:
                label_ids.append(self.entity_labelmapper.get_id("NA"))
            else:
                if last_ind == match_target_index:
                    prefix = "I-"
                else:
                    prefix = "B-"
                if self.model_type == ModelType.BINARY:
                    entity_name = "PRESENT"
                else:
                    entity_name = entities[match_target_index].name
                label_ids.append(self.entity_labelmapper.get_id(prefix + entity_name))
            last_ind = match_target_index
        return label_ids
