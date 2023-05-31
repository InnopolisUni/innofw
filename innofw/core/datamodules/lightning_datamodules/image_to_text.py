from innofw.core.datamodules.lightning_datamodules.base import BaseLightningDataModule
from innofw.constants import Stages
from pathlib import Path
from typing import Any, Optional, Dict
from innofw.core.datasets.image_to_text import ImageToTextDataset
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import os
import re
import PIL.Image as pil
import torch
import sentencepiece as spm


class ImageToTextDatamodule(BaseLightningDataModule):
    task = "image-to-text"

    # @staticmethod
    # def split_sentence(sentence):
    #     return re.findall("\w+", sentence.lower())

    # @staticmethod
    # def encode_sentence(words, sentence, length: int):
    #     sentence = ImageToTextDatamodule.split_sentence(sentence)
    #     sentence = [words.get(x, words.get("<?>")) for x in sentence]
    #     if len(sentence) + 2 > length:
    #         raise ValueError(
    #             f"Unable to encode sentence of size {len(sentence)}\
    #                          in a sentence of target length {length}"
    #         )
    #     sentence = [words.get("<s>")] + sentence + [words.get("</s>")]
    #     while len(sentence) < length:
    #         sentence.append(words.get("<pad>"))
    #     return sentence

    

    def __init__(
        self,
        train: Optional[Dict[str, str]] = None,
        test: Optional[Dict[str, str]] = None,
        infer: Optional[Dict[str, str]] = None,
        stage: Stages = Stages.train,
        vocab_size: int = 65536,
        start_token: int = 1,
        end_token: int = 2,
        pad_token: int = 0,
        # word2int: Dict[str, int] = None,
        max_caption_length: int = 128,
        batch_size: int = 16,
        preprocess: torch.nn.Module = None,
        tokenizer_model: Optional[str] = None,
        augmentations: Dict[str, Any] = None,
        *args,
        **kwargs,
    ):

        super().__init__(train, test, infer=infer, stage=stage, *args, **kwargs)
        # self.word2int = word2int
        self.max_caption_length = max_caption_length
        self.vocab_size = vocab_size

        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.augmentations = augmentations

        if tokenizer_model:
            # Load sentencepiece model
            self.tokenizer_model = spm.SentencePieceProcessor()
            self.tokenizer_model.load(tokenizer_model)
            assert self.tokenizer_model.bos_id() == start_token, "Start token mismatch"
            assert self.tokenizer_model.pad_id() == pad_token, "Pad token mismatch"
            assert self.tokenizer_model.eos_id() == end_token, "End token mismatch"

    def setup_train_test_val(self):
        self.train_df = pd.read_csv(os.path.join(self.train_source, "captions.txt"))

        # Split on test and val
         
        self.val_df = self.train_df.sample(frac=0.2, random_state=42)

        # Remove val from train
        self.train_df = self.train_df.drop(self.val_df.index)
            

        self.train_df['caption'].to_csv(os.path.join(self.train_source, "corpus.txt"), 
                                        index=False, header=False)

        # if self.word2int is None:
        #     self.word2int = dict({"<pad>": 1, "<s>": 2, "</s>": 3, "<?>": 0})
        #     word_set = set()
        #     for sentence in self.train_df["caption"]:
        #         word_set.update(ImageToTextDatamodule.split_sentence(sentence))
        #     self.word2int.update(enumerate(word_set, start=len(self.word2int)))

        if not hasattr(self, 'tokenizer_model'):           
            spm.SentencePieceTrainer.train(
                input=os.path.join(self.train_source, "corpus.txt"),
                model_prefix="tokenizer_",
                vocab_size=self.vocab_size,
                model_type="bpe",
                character_coverage=1.0,
                bos_id=self.start_token,
                pad_id=self.pad_token,
                eos_id=self.end_token,
                unk_id=self.vocab_size - 1,
            )

            self.tokenizer_model = spm.SentencePieceProcessor()
            self.tokenizer_model.load("tokenizer_.model")
    

    def setup_infer(self):
        self.test_df = pd.read_csv(os.path.join(self.test_source, "captions.txt"))

    def val_dataloader(self):
        return DataLoader(
            ImageToTextDataset(
                images_path=self.train_source,
                df=self.val_df,
                # captions_path=os.path.join(self.predict_source, "captions.txt"),
                encoder=self.tokenizer_model,
                transforms=self.augmentations['val'],
                captions_length=self.max_caption_length,
            ), batch_size=self.batch_size
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #          ImageToTextDataset(
    #             image_dir=os.path.join(self.test_source, "Images"),
    #             # captions_path=os.path.join(self.test_source, "captions.txt"),
    #             encoder=self.tokenizer_model,
    #             transforms=self.augmentations['test']
                
    #         ), batch_size=self.batch_size
    #     )
    
    def save_preds(self, preds, stage: Stages, dst_path: Path):
        if stage == Stages.infer:
            preds = pd.DataFrame(preds, columns=["caption"])
            preds.to_csv(dst_path / "captions.txt", index=False)
         

    def train_dataloader(self):
        return DataLoader(
            ImageToTextDataset(
                images_path=self.train_source, 
                # captions_path=os.path.join(self.train_source, "captions.txt"),
                df=self.train_df,
                encoder=self.tokenizer_model,
                transforms=self.augmentations['train'],
                captions_length=self.max_caption_length,
            ), batch_size=self.batch_size
        )
