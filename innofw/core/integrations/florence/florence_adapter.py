import logging
from pathlib import Path

from tqdm import tqdm
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from ..base_integration_models import BaseIntegrationModel
from .utils import draw_bbox_on_image, init_tokenizer, post_process_generation
from .florence_datamodule import FlorenceImageDataModuleAdapter
from innofw.constants import Frameworks, Stages
from innofw.core.models import BaseModelAdapter, register_models_adapter


class FlorenceModel(BaseIntegrationModel):
    framework = Frameworks.florence

    def __init__(self, *args, **kwargs):
        token_path = kwargs["tokenizer_save_path"]
        self.token_path = Path(__file__).resolve().parents[4] / token_path
        self.tokenizer = init_tokenizer(self.token_path)
        self.DEVICE = kwargs["device"]
        self.task_prompt = kwargs["task_prompt"]
        self.text_input = kwargs["text_input"]
        self.ort_session = None

    def generate_text(self, task_prompt, tensor, text_input=None) -> str:
        """

        Args:
            task_prompt:
            tensor:
            text_input:

        Returns:

        """
        tokenizer: AutoTokenizer = self.tokenizer

        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        input_ids_np = tokenizer(prompt, return_tensors="pt")["input_ids"]
        input_ids_np = input_ids_np.cpu().numpy()
        pixel_values_np = tensor.cpu().numpy()

        generated_tokens = self.get_generated_tokens(input_ids_np, pixel_values_np)

        # decode of generated tokens
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        return generated_text

    def get_generated_tokens(self, input_ids_np, pixel_values_np):
        """just in case ask Imran
        generated_tokens = [start_token_id], but we do not HF model to get it implicitly

        comparison is to be next_token_id == model.config.eos_token_id, but we don't have model

        Args:
            input_ids_np:
            pixel_values_np:

        Returns:

        """
        generated_tokens = [0]
        max_length = 60
        for _ in range(max_length):
            decoder_input_ids = torch.tensor([generated_tokens], dtype=torch.long)
            decoder_input_ids_np = decoder_input_ids.cpu().numpy()

            next_token_id = self.run(
                decoder_input_ids_np, input_ids_np, pixel_values_np
            )

            generated_tokens.append(int(next_token_id))

            if next_token_id == 1:
                break
        return generated_tokens

    def run(self, decoder_input_ids, input_ids, pixel_values):
        outputs = self.ort_session.run(
            None,
            {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "decoder_input_ids": decoder_input_ids,
            },
        )
        logits = outputs[0]
        next_token_logits = logits[:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1)[0]
        return next_token_id

    def load_weights(self, ckpt_path):
        self.ort_session = ort.InferenceSession(ckpt_path)

    def predict(self, tensor, size, text_input):
        tokenizer = self.tokenizer
        prompt = self.task_prompt
        generated_text = self.generate_text(prompt, tensor, text_input)
        results = post_process_generation(generated_text, prompt, size, tokenizer)
        return results


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

    def test(self, data, ckpt_path=None):
        raise NotImplementedError

    def predict(self, datamodule: FlorenceImageDataModuleAdapter, ckpt_path=None):
        datamodule.setup()
        if ckpt_path is not None:
            self.model.load_weights(str(ckpt_path))
        ds = datamodule.predict_dataloader()
        pbar = tqdm(ds, total=len(ds))
        for entry in pbar:
            text_input = entry["text_input"] or self.model.text_input
            assert text_input is not None, "Provide text_input, ie cardiomegaly"
            file_name = entry["image_name"]
            pbar.set_description(
                f"Florence inference processing {file_name}:{text_input}"
            )

            size = entry["orig_size"]
            tensor = entry["tensor"]
            results = self.model.predict(tensor, size, text_input)

            image = entry["image"]
            image_with_bbox = draw_bbox_on_image(image, results)
            result = {
                "file_name": file_name,
                "image": image_with_bbox,
                "text_input": text_input,
            }
            datamodule.save_preds(result, stage=Stages.predict, dst_path=self.log_dir)
