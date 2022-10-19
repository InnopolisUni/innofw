import numpy as np


class ActiveDataModule:
    _preprocessed: bool = False

    def __init__(self, datamodule, init_size: float = 0.1):
        self.datamodule = datamodule
        self.init_size = init_size
        self.setup()

    def update_indices(self, indices):
        idx_to_add = self.pool_idxs[indices]
        self.pool_idxs = np.setdiff1d(self.pool_idxs, idx_to_add)
        self.train_idxs = np.concatenate([self.train_idxs, idx_to_add], axis=None)

    def train_dataloader(self):
        return {
            "x": self.train_dl["x"].iloc[self.train_idxs],
            "y": self.train_dl["y"][self.train_idxs],
        }

    def test_dataloader(self):
        return self.datamodule.test_dataloader()

    def pool_dataloader(self):
        return {
            "x": self.train_dl["x"].iloc[self.pool_idxs],
        }

    def setup(self):
        if self._preprocessed:
            return

        self.datamodule.setup()

        self.train_dl = self.datamodule.train_dataloader()
        self.train_idxs = np.random.choice(
            len(self.train_dl["y"]),
            size=int(len(self.train_dl["y"]) * self.init_size),
            replace=False,
        )
        self.pool_idxs = np.setdiff1d(
            np.arange(len(self.train_dl["y"])), self.train_idxs
        )
        self._preprocessed = True


def get_active_datamodule(datamodule):
    return ActiveDataModule(datamodule)
