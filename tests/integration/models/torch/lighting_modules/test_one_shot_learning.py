import torch

from innofw.core.models.torch.lightning_modules import OneShotLearningLightningModule


class OSLLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, out_1, out_2, label):
        return ((out_1 - out_2) ** 2).mean()


class OSLModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Identity()

    def forward(self, x1, x2):
        return self.model(x1), self.model(x2)


def test_osl():
    osl_model = OneShotLearningLightningModule(model=OSLModel(),
                                               losses=[["MSE", 1, OSLLoss()]],
                                               optimizer_cfg=None, scheduler_cfg=None)

    osl_model.training_step(batch=[torch.zeros((100, 100)), torch.zeros((100, 100)), 0],
                            batch_idx=0)

    osl_model.validation_step(batch=[torch.zeros((100, 100)), torch.zeros((100, 100)), 0],
                              batch_idx=0)

    osl_model.predict_step(batch=[torch.zeros((100, 100)), torch.zeros((100, 100))], batch_idx=0)

    osl_model.predict_step(batch=[torch.zeros((2, 100, 100)), torch.zeros((2, 100, 100))],
                           batch_idx=0)