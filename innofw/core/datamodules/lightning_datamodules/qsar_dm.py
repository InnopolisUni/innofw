import logging
from enum import Enum
from os import cpu_count
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import selfies as sf
import torch
from rdkit import rdBase
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from tqdm.contrib.concurrent import process_map

from innofw.constants import Frameworks
from innofw.constants import Stages
from innofw.core.datamodules.lightning_datamodules.base import (
    BaseLightningDataModule,
)

rdBase.DisableLog("rdApp.error")


class WorkMode(Enum):
    VAE = "vae"
    FORWARD = "forward"
    REVERSE = "reverse"


class QsarSelfiesDataModule(BaseLightningDataModule):
    """
    A DataModule class for selfies datasets.

    Attributes
    ----------
    smiles_col : str
        Specify the name of the column containing smiles strings
    target_col : str
        Specify the column name of the target variable in your dataset
    val_size : int
        Specify the fraction of data to be used as validation set
    work_mode : WorkMode
        Specify the type of model we want to use


    Methods
    -------
    setup_train_test_val(**kwargs):
        The setup_train_test_val function is used to split the training data into a train and validation set.
        The function takes in the dataset, smiles_col, target_col and val_size as parameters. The smiles column
        is used to extract all of the SMILES strings from each row of the csv file. The target column is used to
        extract all of the targets for each SMILES string in that row. Both sets are then converted into selfies using
        the self.smiles2selfies function which converts a list of SMILES strings into their corresponding selfies sequences.
    """

    task = [
        "text-vae",
        "qsar-regression",
        "text-vae-reverse",
        "text-vae-forward",
        "text-vae",
    ]
    framework = [Frameworks.torch]

    _preprocessed: bool = False

    def __init__(
        self,
        train,
        test,
        smiles_col: str,
        target_col,
        infer=None,
        val_size: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 1,
        work_mode: WorkMode = WorkMode.VAE,
        *args,
        **kwargs,
    ):
        super().__init__(
            train=train,
            test=test,
            batch_size=batch_size,
            num_workers=num_workers,
            infer=infer,
            *args,
            **kwargs,
        )
        if self.train is None:
            self.train_source = self._get_data(train)
        if self.test is None:
            self.test_source = self._get_data(test)

        self.smiles_col = smiles_col
        self.target_col = target_col
        self.val_size = val_size
        self.work_mode = WorkMode(work_mode)

    def setup_train_test_val(self, **kwargs) -> None:
        train_csv = pd.read_csv(self.train_source)
        test_csv = pd.read_csv(self.test_source)

        train_smiles = train_csv[self.smiles_col].values
        test_smiles = test_csv[self.smiles_col].values

        train_targets = train_csv[self.target_col].values
        test_targets = test_csv[self.target_col].values

        train_selfies = self.smiles2selfies(train_smiles)
        test_selfies = self.smiles2selfies(test_smiles)

        train_selfies, train_targets = zip(
            *filter(
                lambda x: x[0] is not None, zip(train_selfies, train_targets)
            )
        )
        test_selfies, test_targets = zip(
            *filter(
                lambda x: x[0] is not None, zip(test_selfies, test_targets)
            )
        )

        selfies_dataset = np.concatenate((train_selfies, test_selfies))

        alphabet = sf.get_alphabet_from_selfies(selfies_dataset)
        alphabet.add("[nop]")  # [nop] is a special padding symbol
        self.alphabet = list(sorted(alphabet))

        self.pad_to_len = max(sf.len_selfies(s) for s in selfies_dataset)
        self.symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
        self.idx_to_symbol = {i: s for i, s in enumerate(alphabet)}

        val_size = int(len(train_selfies) * self.val_size)
        train_size = len(train_selfies) - val_size
        self.train_selfies_dataset, self.val_selfies_dataset = random_split(
            SelfiesDataset(train_selfies, train_targets),
            (train_size, val_size),
        )
        self.test_selfies_dataset = SelfiesDataset(test_selfies, test_targets)

        self.collator = SelfiesCollator(self.symbol_to_idx, self.pad_to_len)

    def setup_infer(self):
        self.setup_train_test_val()

        predict_csv = pd.read_csv(self.predict_source)
        predict_smiles = predict_csv[self.smiles_col].values
        predict_targets = predict_csv[self.target_col].values
        predict_selfies = self.smiles2selfies(predict_smiles)

        self.predict_indexes, predict_selfies, predict_targets = zip(
            *filter(
                lambda x: x[1] is not None,
                zip(predict_csv.index, predict_selfies, predict_targets),
            )
        )

        self.predict_selfies_dataset = SelfiesDataset(
            predict_selfies, predict_targets
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_selfies_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_selfies_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_selfies_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_selfies_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collator,
        )

    def save_preds(self, preds: List[torch.Tensor], stage: Stages, dst_path: Path):
        if self.work_mode is WorkMode.VAE:
            unrolled_preds: List[List[int]] = [
                pred.argmax(dim=1).tolist()
                for batch in preds
                for pred in batch
            ]
            preds_smiles = self.decode(unrolled_preds)
            preds_series = pd.Series(preds_smiles, index=self.predict_indexes)
            df = pd.read_csv(
                getattr(self, f"{stage.name}_dataset"),
                usecols=[self.smiles_col],
            )

            df["reconstructed_smiles"] = preds_series
            dst_filepath = Path(dst_path) / "preds.csv"
            df.to_csv(dst_filepath, index=False)
            logging.info(f"Saved results to: {dst_filepath}")

        elif self.work_mode is WorkMode.FORWARD:
            unrolled_preds = [pred.item() for batch in preds for pred in batch]
            preds_series = pd.Series(
                unrolled_preds, index=self.predict_indexes
            )
            df = pd.read_csv(
                getattr(self, f"{stage.name}_dataset"),
                usecols=[self.smiles_col],
            )
            if self.target_col is None:
                df["y"] = preds_series
            else:
                df[self.target_col] = preds_series
            dst_filepath = Path(dst_path) / "preds.csv"
            df.to_csv(dst_filepath, index=False)
            logging.info(f"Saved results to: {dst_filepath}")
        elif self.work_mode is WorkMode.REVERSE:
            x_hat_batch, y_hat_batch = zip(*preds)
            x_hats: List[List[int]] = [
                x_hat.argmax(dim=1).tolist()
                for batch in x_hat_batch
                for x_hat in batch
            ]
            y_hat = [y_hat.item() for batch in y_hat_batch for y_hat in batch]

            smiles = self.decode(x_hats)

            df = pd.DataFrame({"generated_smiles": smiles})

            if self.target_col is None:
                df["y"] = y_hat
            else:
                df[self.target_col] = y_hat

            dst_filepath: Path = Path(dst_path) / "preds.csv"
            df.to_csv(dst_filepath, index=False)
            logging.info(f"Saved results to: {dst_filepath}")

    def decode(self, batch_list: List[List[int]]):
        decoded_smiles = []
        for seq in batch_list:
            selfies = sf.encoding_to_selfies(seq, self.idx_to_symbol, enc_type="label")
            decoded_smiles.append(sf.decoder(selfies))
        return decoded_smiles

    @staticmethod
    def try_selfies(smiles: str) -> Optional[str]:
        try:
            selfies = sf.encoder(smiles)
        except sf.EncoderError as e:
            selfies = None
        return selfies

    def smiles2selfies(self, smiles: List[str]):
        return process_map(self.try_selfies,smiles,desc="Converting smiles to selfies...",chunksize=len(smiles) // cpu_count())


class SelfiesDataset(Dataset):
    """
    Dataset with smiles.

    Attributes
    ----------
    selfies : list
        Store the selfies that are to be encoded
    targets : str
        Store the targets for each selfies

    Methods
    -------
    __getitem__(index):
        The __getitem__ function is a special function that allows the class to be indexed.
        For example, if we have a class called &quot;MyClass&quot;, then MyClass()[0] will return the first item in MyClass.
        This is why we can call self.selfies[index], because self refers to an instance of our class.
    """

    def __init__(self, selfies, targets) -> None:
        assert len(selfies) == len(targets)
        self.selfies = selfies
        self.targets = targets

    def __getitem__(self, index):
        return self.selfies[index], self.targets[index]

    def __len__(self):
        return len(self.selfies)


class SelfiesCollator:
    """
    Collator for selfies encoding.

    Attributes
    ----------
    vocab_stoi : dict
        Store the vocabulary
    pad_to_len : int
        Pad the length of a sentence to a certain number

    Methods
    -------
    __call__(data):
        Returns the one hot encoding of the input batch and the labels
    """

    def __init__(self, vocab_stoi, pad_to_len):
        self.vocab_stoi = vocab_stoi
        self.pad_to_len = pad_to_len

    def __call__(self, data):
        selfies_batch, labels_batch = zip(*data)
        hot_batch = []
        for selfies in selfies_batch:
            one_hot = sf.selfies_to_encoding(
                selfies, self.vocab_stoi, self.pad_to_len, enc_type="one_hot"
            )
            hot_batch.append(one_hot)
        return torch.tensor(
            np.asarray(hot_batch), dtype=torch.float
        ), torch.tensor(labels_batch, dtype=torch.float)
