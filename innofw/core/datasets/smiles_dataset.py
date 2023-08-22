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

        X, y = self.calculate_descriptors(
            [AllChem.GetMorganFingerprintAsBitVect, MACCSkeys.GenMACCSKeys],
            smiles,
            property_list,
        )
        self.X = X
        self.y = pd.DataFrame({property_name: y})

    @staticmethod
    def _convert_smiles(smiles: Sequence[str], y: Sequence) -> tuple:
        mols = []
        y_cleaned = []

        for mol, property_, smiles in tqdm(
            zip(smiles, y, smiles),
            desc="Cleaning salts for SMILES...",
            total=len(smiles),
        ):
            mol_cleaned = clean_salts(mol)
            if mol_cleaned is not None:
                mols.append(mol_cleaned)
                y_cleaned.append(property_)

        return mols, y_cleaned

    def calculate_descriptors(
        self, featurizers: List, smiles: Sequence[str], y: Sequence
    ) -> tuple:
        mols, y_cleaned = self._convert_smiles(smiles, y)

        smiles_features = {}

        for featurizer in tqdm(featurizers, desc="Calculating descriptors..."):
            if featurizer == AllChem.GetMorganFingerprintAsBitVect:
                smiles_features[featurizer.__name__] = np.vstack(
                    [featurizer(mol, 2) for mol in mols]
                )
            else:
                smiles_features[featurizer.__name__] = np.vstack(
                    [featurizer(mol) for mol in mols]
                )

        X = np.concatenate(
            [
                smiles_features[featurizer.__name__]
                for featurizer in featurizers
            ],
            axis=1,
        )

        return X, y_cleaned

    def __getitem__(self, idx):
        return self.X[idx], self.y.loc[idx]

    def __len__(self):
        return len(self.y)

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
