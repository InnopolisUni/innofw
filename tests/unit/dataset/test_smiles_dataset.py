import numpy as np
import pandas as pd
from innofw.core.datasets.smiles_dataset import SmilesDataset
from tests.utils import get_test_folder_path


def test_smiles_dataset_smoke():
    # initialize config
    test_csv_path = (
        get_test_folder_path() / "data/tabular/molecular/smiles/qm9/test/test.csv"
    )
    smiles_col = "smiles"
    target_col = "gap"

    # prepare df
    test_df = pd.read_csv(test_csv_path, usecols=[smiles_col, target_col])
    smiles = test_df[smiles_col]
    property_list = test_df[target_col]

    sut = SmilesDataset(
        smiles=smiles, property_list=property_list, property_name=target_col
    )

    # test correct initializing of SmilesDataset
    assert sut is not None
    assert sut.X is not None
    assert sut.y is not None
    assert sut.property_name == target_col
    assert isinstance(sut.X, np.ndarray)
    assert isinstance(sut.y, np.ndarray)
    assert len(sut.X) == len(smiles)
    assert len(sut.y) == len(property_list)

    # test correct sample behavior
    sample = sut[0]
    assert sample is not None
    assert len(sample) == 2
    assert isinstance(sample[0], np.ndarray)
    assert isinstance(sample[1], float)


def test_smiles_dataset_from_df():
    # initialize config
    test_csv_path = (
        get_test_folder_path() / "data/tabular/molecular/smiles/qm9/test/test.csv"
    )
    smiles_col = "smiles"
    target_col = "gap"

    # prepare df
    test_df = pd.read_csv(test_csv_path, usecols=[smiles_col, target_col])

    sut = SmilesDataset.from_df(
        test_df, property_name=target_col, smiles_col=smiles_col
    )

    # test correct initializing of SmilesDataset
    assert sut is not None
    assert sut.X is not None
    assert sut.y is not None
    assert sut.property_name == target_col
    assert isinstance(sut.X, np.ndarray)
    assert isinstance(sut.y, np.ndarray)

    # test correct sample behavior
    sample = sut[0]
    assert sample is not None
    assert len(sample) == 2
    assert isinstance(sample[0], np.ndarray)
    assert isinstance(sample[1], float)


# TODO: Test all public methods
# TODO: Negative tests
