# standard libraries
from typing import Optional, Any

# third party libraries
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
import segmentation_models_pytorch as smp
import hydra
import torch
import hub


class TiledHubSegmentationTestMetricsCalculationCallback(Callback):
    def __init__(
        self,
        metrics,
        stages,
        threshold: float = 0.5,
        label_tensor_name: str = "labels",
        filename_idx_tensor_name: str = "file_idx",
        pad_map_tensor_name: str = "pad_maps",
        *args,
        **kwargs,
    ):
        self.metrics_cfg = metrics

        self.label_tensor_name = label_tensor_name
        self.pad_map_tensor_name = pad_map_tensor_name
        self.threshold = threshold  # todo: is it needed?
        self.state_dict = {"fit": {}, "val": {}, "test": {}}
        self.rounding_digits = 3
        self.stages = stages

        # retrieve file mappings
        self.filename_idx_tensor_name = filename_idx_tensor_name
        self.idx2filename = {}
        self.all_file_indices = {}
        self.tensor_state_dict = {}

        for stage_name, stage_conf in self.stages.items():
            self.idx2filename[stage_name] = {}
            with hub.dataset(stage_conf["file_info_hub_ds"]) as ds:
                # get image_name
                for sample, name in zip(ds[self.filename_idx_tensor_name], ds.filename):
                    idx = sample.data()[0]
                    filename = name.data()

                    self.idx2filename[stage_name][idx] = filename
            self.all_file_indices[stage_name] = self.idx2filename[stage_name].keys()
            self.tensor_state_dict[stage_name] = None

    def on_stage_batch_end(self, pl_module: LightningModule, stage, outputs, batch):
        output = outputs[self.stages[stage].model_out_name]
        label, pad_map = batch[self.label_tensor_name], batch[self.pad_map_tensor_name]

        label = torch.moveaxis(label, -1, 1)
        pad_map = torch.moveaxis(pad_map, -1, 1)

        output *= 1 - pad_map
        label *= 1 - pad_map

        tp, fp, fn, tn = smp.metrics.get_stats(
            output, label, mode="binary", threshold=self.threshold
        )

        indices = batch[self.filename_idx_tensor_name]

        # for label in unique_labels:
        def return_sum(stat_tensor):
            t_sum = torch.zeros(
                (len(self.all_file_indices[stage]), 1), dtype=torch.int64
            ).to(indices.device)
            for i in self.all_file_indices[stage]:
                t_sum[i] += stat_tensor[indices == i].sum()
            return t_sum

        tp_sum = return_sum(tp)
        fp_sum = return_sum(fp)
        fn_sum = return_sum(fn)
        tn_sum = return_sum(tn)
        if self.tensor_state_dict[stage] is None:
            self.tensor_state_dict[stage] = torch.concat(
                [tp_sum, fp_sum, fn_sum, tn_sum], dim=1
            )
        else:
            self.tensor_state_dict[stage] += torch.concat(
                [tp_sum, fp_sum, fn_sum, tn_sum], dim=1
            )

        for metrics_name, metrics_func in self.metrics_cfg.items():
            score = hydra.utils.instantiate(
                metrics_func, torch.sum(tp), torch.sum(fp), torch.sum(tn), torch.sum(fn)
            )

            pl_module.log(
                f"metrics/{stage}/{metrics_name}/",
                round(score.item(), self.rounding_digits),
                on_step=True,
            )

    def on_stage_epoch_end(self, pl_module: LightningModule, stage: str):
        tp = self.tensor_state_dict[stage][:, 0]
        fp = self.tensor_state_dict[stage][:, 1]
        fn = self.tensor_state_dict[stage][:, 2]
        tn = self.tensor_state_dict[stage][:, 3]
        for metrics_name, metrics_func in self.metrics_cfg.items():
            score = hydra.utils.instantiate(metrics_func, tp, fp, tn, fn)

            for key, filename in self.idx2filename[stage].items():
                pl_module.log(
                    f"metrics_per_file/{stage}/{metrics_name}/{filename}",
                    round(score[key].item(), self.rounding_digits),
                )

        self.state_dict[stage] = None

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.on_stage_batch_end(pl_module, "test", outputs, batch)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        self.on_stage_batch_end(pl_module, "fit", outputs, batch)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.on_stage_batch_end(pl_module, "val", outputs, batch)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_stage_epoch_end(pl_module, "test")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_stage_epoch_end(pl_module, "fit")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.on_stage_epoch_end(pl_module, "val")
