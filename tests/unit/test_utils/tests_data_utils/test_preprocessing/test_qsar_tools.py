from typing import Optional
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from innofw.utils.data_utils.preprocessing.qsar_tools import *


class TestCleanSalts:
    def test_clean_salts_with_valid_smiles(self):
        smiles = "CC(=O)O.[Na+]"
        expected_result = Chem.MolFromSmiles("CC(=O)O")
        assert clean_salts(smiles) == expected_result

    def test_clean_salts_with_invalid_smiles(self):
        smiles = "invalid_smiles"
        assert clean_salts(smiles) is None

    def test_clean_salts_with_empty_smiles(self):
        smiles = ""
        assert clean_salts(smiles) is None

    def test_clean_salts_with_none_input(self):
        assert clean_salts(None) is None
