import logging
from pathlib import Path

from tqdm import tqdm

from ..base_integration_models import BaseIntegrationModel
from .utils import (
    draw_bbox_on_image,
    init_tokenizer,
    generate_text,
    post_process_generation,
)
from .florence_datamodule import FlorenceDataModuleAdapter
from innofw.constants import Frameworks, Stages
from innofw.core.models import BaseModelAdapter, register_models_adapter


class FlorenceModel(BaseIntegrationModel):
    framework = Frameworks.florence

    def __init__(self, *args, **kwargs):
        token_path = kwargs["tokenizer_save_path"]
        self.token_path = Path().resolve().parent / token_path
        self.tokenizer = init_tokenizer(self.token_path)
        self.DEVICE = kwargs["device"]
        self.task_prompt = kwargs["task_prompt"]
        self.text_input = kwargs["text_input"]


@register_models_adapter("florence_adapter")
class FlorenceModelAdapter(BaseModelAdapter):
    @staticmethod
    def is_suitable_model(model) -> bool:
        return isinstance(model, FlorenceModel)

    def _test(self, data):
        pass

    def _train(self, data):
        pass

    def _predict(self, data):
        pass

    framework = Frameworks.florence

    def __init__(
        self,
        model,
        log_dir,
        trainer_cfg,
        *args,
        **kwargs,
    ):
        super().__init__(model, log_dir)

    def train(self, data, ckpt_path=None):
        raise NotImplementedError

    def test(self, data, ckpt_path=None, flags=""):
        raise NotImplementedError

    def run_florence(self, tensor, size, text_input, ckpt_path: str):
        tokenizer = self.model.tokenizer
        prompt = self.model.task_prompt
        generated_text = generate_text(
            prompt, tensor, ckpt_path, tokenizer, text_input, self.model.DEVICE
        )
        results = post_process_generation(generated_text, prompt, size, tokenizer)
        return results

    def predict(self, dataset: FlorenceDataModuleAdapter, ckpt_path=None):
        dataset.setup()
        ds = dataset.predict_dataloader()
        total = ds.size
        pbar = tqdm(ds, desc="Florence inference", total=total)
        for entry in pbar:
            image = entry["image"]
            text_input = entry["text_input"] or self.model.text_input
            assert text_input is not None, "Provide text_input, ie cardiomegaly"
            file_name = entry["image_name"]
            size = entry["orig_size"]
            tensor = entry["tensor"]
            pbar.set_description(f"Processing {file_name}:{text_input}")

            results = self.run_florence(tensor, size, text_input, str(ckpt_path))
            modified_image = draw_bbox_on_image(image, results)
            result = {
                "file_name": file_name,
                "image": modified_image,
                "text_input": text_input,
            }
            dataset.save_preds(result, stage=Stages.predict, dst_path=self.log_dir)
