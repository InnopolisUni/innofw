from typing import Optional
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from unittest import TestCase, mock
from unittest.mock import patch

from innofw.utils.data_utils.preprocessing.qsar_tools import *


class TestCleanSalts(TestCase):

    def test_clean_salts_returns_none_for_invalid_input(self):
        self.assertIsNone(clean_salts('invalid_smiles'))

    @patch('rdkit.Chem.MolFromSmiles')
    @patch.object(SaltRemover, 'StripMol')
    def test_clean_salts_returns_expected_output(self, mock_stripmol, mock_molfromsmiles):
        mock_molfromsmiles.return_value = 'valid_mol'
        mock_stripmol.return_value = 'clean_mol'
        expected_output = 'clean_mol'
        actual_output = clean_salts('valid_smiles')
        self.assertEqual(expected_output, actual_output)
