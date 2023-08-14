import logging
from numbers import Number
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from torch.utils.data import Dataset
from tqdm import tqdm

from innofw.utils.data_utils.preprocessing import clean_salts

logging.getLogger("rdkit").propagate = False


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

    generate_descriptors(self, featurizers: List[rdkit.Chem.rdMolDescriptors.MolecularDescriptor]):
        creates descriptions out of featurizers
    init_features(self, features: Optional[List[str]] = None):
        initialize X-features
    from_df(cls, df: pd.DataFrame, property_name: str, smiles_col: str = "smiles", property_col: Optional[str] = None):
        initializes class object using data frame

    """

    def __init__(
        self,
        smiles: Sequence[str],
        property_list: Sequence[Number],
        property_name: str,
    ):
        self.smiles = smiles
        self.y = np.array(property_list)
        self.property_name = property_name

        self._convert_smiles()

        self.generate_descriptors(
            [AllChem.GetMorganFingerprintAsBitVect, MACCSkeys.GenMACCSKeys]
        )

    def _convert_smiles(self):
        self.mols = []
        self.smiles_cleaned = []
        self.y_cleaned = []

        for mol, property_, smiles in tqdm(
            zip(self.smiles, self.y, self.smiles),
            desc="Cleaning salts...",
            total=len(self.smiles),
        ):
            mol_cleaned = clean_salts(mol)
            if mol_cleaned is not None:
                self.mols.append(mol_cleaned)
                self.y_cleaned.append(property_)
                self.smiles_cleaned.append(smiles)

        self.y = np.array(self.y_cleaned)
        self.smiles = self.smiles_cleaned

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

    def generate_descriptors(self, featurizers):
        self.smiles_features = {}
        self.featurizer_names = []

        for featurizer in tqdm(featurizers, desc="Calculating descriptors..."):
            if featurizer == AllChem.GetMorganFingerprintAsBitVect:
                self.smiles_features[featurizer.__name__] = np.vstack(
                    [featurizer(mol, 2) for mol in self.mols]
                )
            else:
                self.smiles_features[featurizer.__name__] = np.vstack(
                    [featurizer(mol) for mol in self.mols]
                )
            self.featurizer_names.append(featurizer.__name__)

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
