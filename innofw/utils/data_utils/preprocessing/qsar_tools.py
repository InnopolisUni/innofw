from typing import Optional

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


def clean_salts(smiles: str) -> Optional[Chem.rdchem.Mol]:
    remover = SaltRemover(defnFormat="smarts")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    res = remover.StripMol(mol)
    return res
