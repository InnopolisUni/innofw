# from typing import Any
# from typing import Optional
#
# import hydra
# import segmentation_models_pytorch as smp
# import torch
# from pytorch_lightning.callbacks import Callback
#
#
# class LoggingSMPMetricsCallback(Callback):
#     def __init__(self, metrics, log_every_n_steps=50, *args, **kwargs):
#         self.metrics = metrics
#         self.train_scores = {}
#         self.val_scores = {}
#         self.test_scores = {}
#         self.log_every_n_steps = log_every_n_steps
#
#     def on_stage_batch_end(self, outputs, batch, dict_to_store):
#         masks = batch["labels"].long().squeeze()
#         logits = outputs["logits"].squeeze()
#         threshold = 0.4
#         tp, fp, fn, tn = smp.metrics.get_stats(
#             logits, masks, mode="binary", threshold=threshold
#         )
#
#         def compute_func(func):
#             res = hydra.utils.instantiate(
#                 func, tp, fp, fn, tn, reduction="micro"
#             ).item()
#             res = round(res, 3)
#             return res
#
#         for name, func in self.metrics.items():
#             score = compute_func(func)
#
#             if name not in dict_to_store:
#                 dict_to_store[name] = [score]
#             else:
#                 dict_to_store[name].append(score)
#
#     def on_stage_epoch_end(
#         self, trainer, pl_module: "pl.LightningModule", dict_with_scores
#     ):
#         for name, scores in dict_with_scores.items():
#             tensorboard = pl_module.logger.experiment
#             mean_score = torch.mean(torch.tensor(scores))
#             # for logger in trainer.loggers:
#             #     logger.add_scalar(f"metrics/train/{name}", mean_score,
#             #                       step=trainer.fit_loop.epoch_loop.batch_idx)
#             tensorboard.add_scalar(
#                 f"metrics/train/{name}", mean_score, trainer.current_epoch
#             )
#
#     def on_train_batch_end(
#         self,
#         trainer: "pl.Trainer",
#         pl_module: "pl.LightningModule",
#         outputs,  # : STEP_OUTPUT
#         batch: Any,
#         batch_idx: int,
#         unused: Optional[int] = 0,
#     ) -> None:
#         if (batch_idx + 1) % self.log_every_n_steps == 0:
#             return
#
#         self.on_stage_batch_end(outputs, batch, self.train_scores)
#
#     def on_validation_batch_end(
#         self,
#         trainer: "pl.Trainer",
#         pl_module: "pl.LightningModule",
#         outputs,
#         batch: Any,
#         batch_idx: int,
#         dataloader_idx: int,
#     ) -> None:
#         if (batch_idx + 1) % self.log_every_n_steps == 0:
#             return
#
#         self.on_stage_batch_end(outputs, batch, self.val_scores)
#
#     def on_train_epoch_end(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
#     ) -> None:
#         self.on_stage_epoch_end(trainer, pl_module, self.train_scores)
#
#     def on_validation_epoch_end(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
#     ) -> None:
#         self.on_stage_epoch_end(trainer, pl_module, self.val_scores)
