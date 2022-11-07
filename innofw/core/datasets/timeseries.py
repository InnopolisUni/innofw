import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np


class ECGDataset(Dataset):
    """
        A class to represent a custom ECG Dataset.

        file_name: str
            path to csv file with ECG train data

        Methods
        -------
        __getitem__(self, idx):
            returns X-features, and Y-targets
    """

    def __init__(self, file_name):
        df = pd.read_csv(file_name)

        x = df.iloc[:, 1:-1].values
        # sequences = x.astype(np.float32).to_numpy().tolist()
        # dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]

        y = df.iloc[:, -1].values

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
