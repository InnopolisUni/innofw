import logging
import pathlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import datasets
import pandas as pd
import torch
from innofw.constants import Frameworks, ModelType, Stages
from innofw.core.datamodules.lightning_datamodules.base import \
    BaseLightningDataModule
from innofw.exceptions import NonUniqueException
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

datasets.disable_caching()


class DrugprotDataModule(BaseLightningDataModule):
    """
    DataModule for Drugprot dataset.

    Attributes
    ----------
    train : str
        path to train dir
    test : str
        path to test dir
    tokenizer : PreTrainedTokenizerBase
        instance of PreTrainedTokenizerBase
    entity_labelmapper: LabelMapper
        label2int mapper
    

    Methods
    -------
    setup_train_test_val():
        Create train test val split.
    """
    task: List[str] = ["text-ner"]
    framework: List[str] = [Frameworks.torch]

    def __init__(
        self,
        train,
        test,
        tokenizer: PreTrainedTokenizerBase,
        entity_labelmapper: "LabelMapper",
        infer=None,
        val_size: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 1,
        random_seed: int = 42,
        stage=None,
        augmentations=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            train=train,
            test=test,
            batch_size=batch_size,
            num_workers=num_workers,
            infer=infer,
            stage=stage,
            *args,
            **kwargs,
        )
        self.tokenizer = tokenizer
        self.entity_labelmapper = entity_labelmapper
        self.model_type = (
            ModelType.BINARY
            if len(self.entity_labelmapper) == 2
            else ModelType.MULTICLASS
        )

        self.val_size = val_size
        self.random_seed = random_seed

    def setup_train_test_val(self, **kwargs) -> None:
        self.train_dataset_raw = datasets.load_from_disk(self.train_dataset)
        self.test_dataset_raw = datasets.load_from_disk(self.test_dataset)

        if isinstance(self.train_dataset_raw, datasets.DatasetDict):
            self.train_dataset_raw = self.train_dataset_raw["train"]
        if isinstance(self.test_dataset_raw, datasets.DatasetDict):
            self.test_dataset_raw = self.test_dataset_raw["test"]

        train_dataset = self.train_dataset_raw
        self.test_dataset = self.test_dataset_raw

        splits = train_dataset.train_test_split(test_size=self.val_size, shuffle=True)
        self.train_dataset = splits["train"]
        self.val_dataset = splits["test"]

        self.collator = DataCollatorWithPaddingAndTruncation(
            max_length=512,
            sequence_keys=["input_ids", "labels"],
        )
        logging.info(self.train_dataset_raw)

    def setup_infer(self):
        print(self.predict_dataset)
        self.predict_dataset_raw = datasets.load_from_disk(self.predict_dataset)

        if isinstance(self.predict_dataset_raw, datasets.DatasetDict):
            self.predict_dataset_raw = self.predict_dataset_raw["test"]

        self.collator = DataCollatorWithPaddingAndTruncation(
            max_length=512,
            sequence_keys=["input_ids", "labels"],
        )
        logging.info(self.predict_dataset_raw)

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset.with_format("pt"),
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset.with_format("pt"),
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset.with_format("pt"),
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )
        return test_dataloader

    def predict_dataloader(self):
        test_dataloader = DataLoader(
            self.predict_dataset_raw.with_format("pt"),
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )
        return test_dataloader

    def save_preds(self, preds, stage: Stages, dst_path: pathlib.Path):
        unrolled_preds = [pred.tolist() for batch in preds for pred in batch]
        df = pd.DataFrame({"prediction": unrolled_preds})
        df.to_json(Path(dst_path) / "preds.json", index=False, orient="table")

    def _prepare(
        self, input_dataset: Union[datasets.Dataset, datasets.DatasetDict]
    ) -> Union[datasets.Dataset, datasets.DatasetDict]:
        def batch_map(rows):
            text_list = rows["text"]
            labels_list = rows["labels"]
            spans_list = rows["spans"]

            entities_list: List[List[NamedEntity]] = [
                sorted(
                    [
                        NamedEntity(name=label, span=span)
                        for label, span in zip(labels, spans)
                    ],
                    key=lambda x: x.span[0],
                )
                for (labels, spans) in zip(labels_list, spans_list)
            ]
            prepared = self._batch_prepare(
                text_list=text_list,
                entities_list=entities_list,
                ignore_non_unique=True,
            )
            return {
                "input_ids": prepared["input_ids"],
                "labels": prepared["labels"],
            }

        prepared_dataset = datasets_exclusive_map(
            input_dataset,
            batch_map,
            batched=True,
            batch_size=10_000,
        )

        return prepared_dataset

    def _batch_prepare(
        self,
        text_list: List[str],
        entities_list: Optional[List[List["NamedEntity"]]] = None,
        flatten=False,
        ignore_non_unique=False,
    ) -> Union[Dict[str, List[List[int]]], List[Dict[str, List[int]]]]:
        encoding = self.tokenizer(
            text_list,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        offset_mapping_list = encoding["offset_mapping"]
        input_ids_list = encoding["input_ids"]

        label_ids_list = []
        if entities_list:
            for spans, entities in zip(offset_mapping_list, entities_list):
                target_spans = [entity.span for entity in entities]
                try:
                    match_target_indexes = self.match_spans(spans, target_spans)
                    label_ids = self._get_label_ids_by_matched_target(
                        match_target_indexes, entities
                    )
                except NonUniqueException as e:
                    if ignore_non_unique:
                        label_ids = None
                    else:
                        raise e
                label_ids_list.append(label_ids)

        if ignore_non_unique and entities_list is not None:
            input_ids_list = [
                input_ids
                for input_ids, label_ids in zip(input_ids_list, label_ids_list)
                if label_ids is not None
            ]
            label_ids_list = [
                label_ids for label_ids in label_ids_list if label_ids is not None
            ]

        result = {
            "input_ids": input_ids_list,
            "offset_mapping": offset_mapping_list,
        }
        if entities_list is not None:
            result["labels"] = label_ids_list

        if flatten:
            result = [
                {key: result[key][index] for key in result}
                for index in range(len(result["input_ids"]))
            ]

        return result

    @staticmethod
    def match_spans(
        spans: Sequence[Tuple[int, int]], target_spans: Sequence[Tuple[int, int]]
    ) -> List[Optional[int]]:
        span_index = 0
        target_span_index = 0
        target_indexes: List[Optional[int]] = [None for _ in spans]
        while span_index < len(spans) and target_span_index < len(target_spans):
            span = spans[span_index]
            target_span = target_spans[target_span_index]
            if span[0] <= target_span[0]:
                if span[1] > target_span[0]:
                    if target_indexes[span_index] is None:
                        target_indexes[span_index] = target_span_index
                    else:
                        raise NonUniqueException(
                            "Target span cannot be assigned uniquely"
                        )
            else:
                if target_span[1] > span[0]:
                    if target_indexes[span_index] is None:
                        target_indexes[span_index] = target_span_index
                    else:
                        raise NonUniqueException(
                            "Target span cannot be assigned uniquely"
                        )
            if span[1] <= target_span[1]:
                span_index += 1
            else:
                target_span_index += 1

        return target_indexes

    def _get_label_ids_by_matched_target(self, match_target_indexes, entities):
        if self.model_type == ModelType.MULTICLASS:
            return [
                self.entity_labelmapper.get_id("NA")
                if match_target_index is None
                else self.entity_labelmapper.get_id(entities[match_target_index].name)
                for match_target_index in match_target_indexes
            ]
        elif self.model_type == ModelType.BINARY:
            return [
                self.entity_labelmapper.get_id("NA")
                if match_target_index is None
                else self.entity_labelmapper.get_id("PRESENT")
                for match_target_index in match_target_indexes
            ]
        else:
            raise ValueError(f"ModelType {self.model_type} not supported")


def datasets_exclusive_map(
    dataset: Union[datasets.Dataset, datasets.DatasetDict], function, *args, **kwargs
):
    if isinstance(dataset, datasets.Dataset):
        columns = dataset.column_names
        mapped_dataset = dataset.map(function, *args, remove_columns=columns, **kwargs)
    elif isinstance(dataset, datasets.DatasetDict):
        mapped_dataset = datasets.DatasetDict()
        for split_name in dataset:
            dataset_split = dataset[split_name]
            columns = dataset_split.column_names
            mapped_dataset[split_name] = dataset_split.map(
                function, *args, remove_columns=columns, **kwargs
            )
    else:
        raise ValueError(f"Not supported dataset type: {type(dataset)}")

    return mapped_dataset


class DataCollatorWithPaddingAndTruncation:
    """
    The data collator with padding and truncation .

    Attributes
    ----------
    max_length : int
        Define the maximum length of a sequence that will be passed to the model
    sequence_keys : list
        Specify which columns in the dataset are sequences
    float_keys : list
        Specify which columns should be treated as float values
    pad_token_id : int
        Specify the token id that will be used for padding

    Methods
    -------
    collate_list_of_dicts(data):
        Collate list of dicts.
    """
    def __init__(self, max_length, sequence_keys=[], float_keys=[], pad_token_id=0):
        self.max_length = max_length
        self.sequence_keys = set(sequence_keys)
        self.pad_token_id = pad_token_id
        self.float_keys = set(float_keys)

    def __call__(self, data):
        return self.collate_list_of_dicts(data)

    def collate_list_of_dicts(self, data):
        any_key = next(iter(self.sequence_keys))
        batch_size = len(data)
        max_len = 0
        for instance in data:
            max_len = max(len(instance[any_key]), max_len)
        max_len = min(max_len, self.max_length)
        result = {
            key: torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.int64)
            for key in self.sequence_keys
        }
        for key in self.sequence_keys:
            for batch_index, instance in enumerate(data):
                current_len = min(max_len, len(instance[key]))
                result[key][batch_index, :current_len] = instance[key][:current_len]

        rest_keys = next(iter(data)).keys() - self.sequence_keys
        for key in rest_keys:
            values = [row[key] for row in data]
            result[key] = torch.stack(values)

        for float_key in self.float_keys:
            if float_key in result:
                result[float_key] = result[float_key].float()

        return result

    def __get_key_type(self, key):
        if key in self.float_keys:
            return torch.float32
        else:
            return torch.int64

    def collate_dict_of_lists(self, data):
        any_key = next(iter(self.sequence_keys))
        batch_size = len(data[any_key])
        max_len = 0
        for instance in data[any_key]:
            max_len = max(len(instance), max_len)
        max_len = min(max_len, self.max_length)
        result = {
            key: torch.full(
                (batch_size, max_len), self.pad_token_id, dtype=self.__get_key_type(key)
            )
            for key in self.sequence_keys
        }
        for key in self.sequence_keys:
            for batch_index, instance in enumerate(data[key]):
                current_len = min(max_len, len(instance))
                for pos_index in range(current_len):
                    result[key][batch_index, pos_index] = instance[pos_index]

        rest_keys = data.keys() - self.sequence_keys

        for key in rest_keys:
            values = data[key]
            result[key] = torch.Tensor(values).type(self.__get_key_type(key))

        return result


@dataclass
class NamedEntity:
    """
    A class to represent a person.

    Attributes
    ----------
    name : str
        name of the entiry
    span : Tuple[int, int]
        slice indexes of text
    """
    name: str
    span: Tuple[int, int]


class LabelMapper:
    """
    The label mapper class.

    Attributes
    ----------
    __name_to_id : Dict[str, int]
        name to id mapping
    __id_to_name : Dict[int, id]
        id to name mapping

    Methods
    -------
    keys(additional=""):
        Returns keys of __name_to_id.
    """

    def __init__(self, label_dict) -> None:
        self.__name_to_id = label_dict.copy()
        self.__id_to_name = {
            label_id: label_name for label_name, label_id in label_dict.items()
        }

    def __len__(self):
        return len(self.__name_to_id)

    def __repr__(self):
        return f"""
LabelMapper(
    {self.__name_to_id}
)
"""

    def keys(self):
        return self.__name_to_id.keys()

    def get_id(self, label_name):
        return self.__name_to_id[label_name]

    def get_name(self, label_id):
        return self.__id_to_name[label_id]

    def to_pickle(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_iterable(cls, label_list: Iterable):
        label_dict = {}
        index = 0
        for label in label_list:
            if label not in label_dict:
                label_dict[label] = index
                index += 1
        return cls(label_dict)

    @classmethod
    def from_pickle(cls, path):
        with open(path, "rb") as f:
            loaded = pickle.load(f)

        if type(loaded) == cls:
            return loaded
        else:
            raise ValueError("Given pickle path contain incorrect class")

    @classmethod
    def from_dict(cls, dict):
        return cls(dict)
