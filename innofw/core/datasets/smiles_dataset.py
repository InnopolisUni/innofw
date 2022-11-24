import logging
from multiprocessing import Pool, cpu_count
from numbers import Number
from typing import List, Optional, Sequence

import deepchem as dc
import numpy as np
import pandas as pd
from innofw.utils.data_utils.preprocessing import clean_salts
from torch.utils.data import Dataset
from tqdm import tqdm

logging.getLogger("deepchem").propagate = False


class SmilesDataset(Dataset):
    """
        A class to represent SMILES Dataset.
        https://www.kaggle.com/c/smiles/data

        smiles: Sequence[str]
        property_list: Sequence[Number]
        property_name: str


        Methods
        -------
        __getitem__(self, idx):
            returns X - features and Y - targets

       generate_descriptors(self, featurizers: List[dc.feat.MolecularFeaturizer]):
            creates descriptions out of featurizers
       init_features(self, features: Optional[List[str]] = None):
            initialize X-features
       from_df(cls, df: pd.DataFrame, property_name: str, smiles_col: str = "smiles", property_col: Optional[str] = None):
            initializes class object using data frame

    """

    cf_featurizer = dc.feat.CircularFingerprint(size=1024)
    maccs_descriptor = dc.feat.MACCSKeysFingerprint()

    def __init__(
        self, smiles: Sequence[str], property_list: Sequence[Number], property_name: str
    ):
        self.smiles = smiles
        self.y = np.array(property_list)
        self.property_name = property_name

        self._convert_smiles()

        self.generate_descriptors([self.cf_featurizer, self.maccs_descriptor])

    def _convert_smiles(self):
        with Pool(cpu_count()) as pool:
            pre_clean = tqdm(
                zip(
                    pool.map(clean_salts, self.smiles), self.y, self.smiles
                ),
                desc="Cleaning salts...",
                total=len(self.smiles),
            )

        self.mols, self.y, self.smiles = zip(
            *(
                (mol, property_, smiles)
                for mol, property_, smiles in pre_clean
                if mol is not None
            )
        )
        self.y = np.array(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def generate_descriptors(self, featurizers: List[dc.feat.MolecularFeaturizer]):
        self.smiles_features = {}
        self.featurizer_names = []
        with Pool(cpu_count()) as pool:
            for featurizer in tqdm(featurizers, desc="Calculating descriptors..."):
                self.smiles_features[type(featurizer).__name__] = np.vstack(
                    pool.map(featurizer.featurize, self.mols)
                )
                self.featurizer_names.append(type(featurizer).__name__)
        self.init_features(self.featurizer_names)

    def init_features(self, features: Optional[List[str]] = None):
        self.cur_features = features or self.cur_features
        X = [
            self.smiles_features[featurizer_name]
            for featurizer_name in self.cur_features
        ]
        self.X = np.concatenate(X, axis=1)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        property_name: str,
        smiles_col: str = "smiles",
        property_col: Optional[str] = None,
    ):
        if property_col is None:
            property_col = property_name
        return cls(df[smiles_col], df[property_col], property_name)
