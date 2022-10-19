from typing import Tuple

import pytorch_lightning as pl
import selfies as sf
import torch
from innofw.core.models.torch.architectures.autoencoders.vae import Encoder, GRUDecoder
from innofw.core.models.torch.lightning_modules.base import BaseLightningModule
from torch import nn


class ChemistryVAELightningModule(BaseLightningModule):
    def __init__(
        self, model: dict, losses, optimizer_cfg, scheduler_cfg, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoder: Encoder = model["encoder"]
        self.decoder: GRUDecoder = model["decoder"]

        self.losses = losses
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.fc_mu = nn.Linear(self.encoder.enc_out_dim, self.decoder.latent_dimension)
        self.fc_var = nn.Linear(self.encoder.enc_out_dim, self.decoder.latent_dimension)

        self.MLP = nn.Sequential(
            nn.Linear(self.decoder.latent_dimension, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, batch: torch.Tensor):
        x = self.encoder(batch.flatten(1))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)

        return self._decode(z, batch)

    def _run_step(self, batch: torch.Tensor):
        x = self.encoder(batch.flatten(1))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self._decode(z, batch), p, q, x

    def _decode(self, z, x):
        out_one_hot = torch.zeros_like(x)
        hide = None
        for seq_index in range(x.size(1)):
            out_one_hot_line, hide = self.decoder(z, hide)
            out_one_hot[:, seq_index, :] = out_one_hot_line[0]
        return out_one_hot

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, x: torch.Tensor):
        z, x_hat, p, q, x = self._run_step(x)
        y_hat = self.MLP(x)
        return z, x_hat, p, q, y_hat

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        z, x_hat, p, q, y_hat = self.step(x)
        loss = self._calc_losses(x, x_hat, p, q, y, y_hat)

        return {"loss": loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch
        z, x_hat, p, q, y_hat = self.step(x)
        loss = self._calc_losses(x, x_hat, p, q, y, y_hat)

        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx, **kwargs):
        x, _ = batch
        x_hat = self(x)
        return x_hat

    def _calc_losses(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        p: torch.Tensor,
        q: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> torch.FloatTensor:
        """Function to compute losses"""
        total_loss = torch.zeros(1, device=self.device, dtype=x.dtype)
        for loss_name, weight, loss in self.losses:
            if loss_name == "mse":
                ls = loss(x, x_hat)
            if loss_name == "target_loss":
                ls = loss(y, y_hat.view_as(y))
            elif loss_name == "kld":
                ls = loss(q, p)
            else:
                continue

            total_loss += weight * ls

        return total_loss

    def configure_optimizers(self):
        """Function to set up optimizers and schedulers"""
        # get all trainable model parameters
        params = [x for x in self.parameters() if x.requires_grad]

        # instantiate models from configurations
        optim = self.optimizer_cfg(params=params)
        # optim = hydra.utils.instantiate(self.optimizer_cfg, params=params)

        # instantiate scheduler from configurations
        try:
            scheduler = self.scheduler_cfg(optim)
            # scheduler = hydra.utils.instantiate(self.scheduler_cfg, optim)
            # return optimizers and schedulers
            return [optim], [scheduler]
        except:
            return [optim]


class ChemistryVAEForwardLightningModule(ChemistryVAELightningModule):
    def forward(self, batch: torch.Tensor):
        x = self.encoder(batch.flatten(1))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        y_hat = self.MLP(z)

        return y_hat


class ChemistryVAEReverseLightningModule(ChemistryVAELightningModule):
    def predict_step(self, batch, batch_idx, **kwargs):
        x_hat, y_hat = self(batch)
        return x_hat, y_hat

    def forward(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        y = y.view(-1, 1)
        latent_dim = self.decoder.latent_dimension
        n_tries = 100

        mu = torch.zeros(latent_dim, dtype=y.dtype, device=self.device)
        std = torch.ones_like(mu)
        p = torch.distributions.Normal(mu, std)

        zs = p.rsample((y.size(0), n_tries))
        y_hats = self.MLP(zs)

        min_abs_diff_idx = (y_hats - y[:, None, :]).abs().argmin(1)
        z = zs[torch.arange(y.size(0)), min_abs_diff_idx.flatten(), :]
        y_hat = y_hats[torch.arange(y.size(0)), min_abs_diff_idx.flatten(), :]

        x_hat = self._decode(z, x)

        return x_hat, y_hat
