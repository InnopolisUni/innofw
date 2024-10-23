import os
import logging
from pathlib import Path

from ..base_integration_models import BaseIntegrationModel
from .utils import draw_bbox_on_image, init_tokenizer, run_florence
from innofw.constants import Frameworks, Stages
from innofw.core.models import BaseModelAdapter, register_models_adapter


class FlorModel(BaseIntegrationModel):
    framework = Frameworks.flor

    def __init__(self, *args, **kwargs):
        token_path = kwargs["tokenizer_save_path"]
        self.token_path = Path().resolve().parent / token_path
        self.tokenizer = init_tokenizer(self.token_path)
        self.DEVICE = kwargs["device"]
        self.task_prompt = kwargs["task_prompt"]


@register_models_adapter("flor_adapter")
class FlorModelAdapter(BaseModelAdapter):
    @staticmethod
    def is_suitable_model(model) -> bool:
        return isinstance(model, FlorModel)

    def _test(self, data):
        pass

    def _train(self, data):
        pass

    def _predict(self, data):
        pass

    framework = Frameworks.flor

    def __init__(
        self,
        model,
        log_dir,
        trainer_cfg,
        *args,
        **kwargs,
    ):
        super().__init__(model, log_dir)

    def update_configs(self, processed_data_path: str):
        pass

    def train(self, data, ckpt_path=None):
        raise NotImplementedError

    def test(self, data, ckpt_path=None, flags=""):
        raise NotImplementedError

    def predict(self, datamodule, ckpt_path=None):
        datamodule.setup()
        ds = datamodule.predict_dataloader()
        from tqdm import tqdm
        pbar = tqdm(ds, desc="Florence inference")
        for entry in pbar:
            image = entry["image"]
            text_input = entry["text_input"]
            file_name = entry["image_name"]
            size = entry["orig_size"]
            tensor = entry["tensor"]
            pbar.set_description(f"Processing {file_name}:{text_input}" )

            results = run_florence(
                self.model.task_prompt,
                tensor,
                size,
                text_input,
                self.model.tokenizer,
                str(ckpt_path),
                self.model.DEVICE,
            )
            modified_image = draw_bbox_on_image(image, results)
            datamodule.save_preds(
                {
                    "file_name": file_name,
                    "image": modified_image,
                    "text_input": text_input,
                },
                stage=Stages.predict,
                dst_path=self.log_dir,
            )
