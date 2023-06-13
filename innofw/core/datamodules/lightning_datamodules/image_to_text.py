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
    """Class defines dataset preparation and dataloader creation for image classification"""

    task = "image-to-text"

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
        max_caption_length: int = 128,
        batch_size: int = 16,
        tokenizer_model: Optional[str] = None,
        augmentations: Dict[str, Any] = None,
        *args,
        **kwargs,
    ):
        """Initialize the datamodule

        Args:
            train (Optional[Dict[str, str]], optional): Dictionary with train data. Defaults to None.
            test (Optional[Dict[str, str]], optional): Dictionary with test data. Defaults to None.
            infer (Optional[Dict[str, str]], optional): Dictionary with inference data. Defaults to None.
            stage (Stages, optional): Stage the datamodule is initialized for. Defaults to Stages.train.
            vocab_size (int, optional): Size of the vocabulary. Defaults to 65536.
            start_token (int, optional): Start token. Defaults to 1.
            end_token (int, optional): End token. Defaults to 2.
            pad_token (int, optional): Pad token. Defaults to 0.
            max_caption_length (int, optional): Max caption length. Defaults to 128.
            batch_size (int, optional): Batch size. Defaults to 16.
            tokenizer_model (Optional[str], optional): Path to tokenizer model. Defaults to None.
            augmentations (Dict[str, Any], optional): Augmentations. Defaults to None.
        """
            
        
        super().__init__(train, test, infer=infer, stage=stage, *args, **kwargs)
        # self.word2int = word2int
        self.max_caption_length = max_caption_length
        self.vocab_size = vocab_size

        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
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
        """Prepare train, test and validation datasets.
        
        The dataset consists of a folder with images and a captions.txt file with captions for each image.
        The captions.txt file has the following format:

        image,caption
        1001773457_577c3a7d70.jpg,A black dog and a spotted dog are fighting
        1001773457_577c3a7d70.jpg,A black dog and a tri-colored dog playing with each other on the road .
        ...

        The dataset is split on train and val. The test dataset is not used for training.
        """


        self.train_df = pd.read_csv(os.path.join(self.train_source, "captions.txt"))

        # Split on test and val
         
        self.val_df = self.train_df.sample(frac=0.2, random_state=42)

        # Remove val from train
        self.train_df = self.train_df.drop(self.val_df.index)
        
        if not hasattr(self, 'tokenizer_model'):  # if no model is provided, train tokenizer        
            # Preparing corpus for tokenizer training
            self.train_df['caption'].to_csv(os.path.join(self.train_source, "corpus.txt"), 
                                            index=False, header=False)
            
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
        """Prepare inference dataset."""
        self.test_df = pd.read_csv(os.path.join(self.predict_source, "captions.txt")) 
        self.tokenizer_model = spm.SentencePieceProcessor()
        self.tokenizer_model.load("tokenizer_.model")
        

    def val_dataloader(self):
        return DataLoader(
            ImageToTextDataset(
                images_path=self.train_source,
                df=self.val_df,
                encoder=self.tokenizer_model,
                transforms=self.augmentations['val'],
                captions_length=self.max_caption_length,
            ), batch_size=self.batch_size
        )
    
    
    def predict_dataloader(self):
        return DataLoader(
            ImageToTextDataset(
                images_path=self.predict_source,
                df=self.test_df,
                encoder=self.tokenizer_model,
                transforms=self.augmentations['test'],
                captions_length=self.max_caption_length,
            ), batch_size=self.batch_size
        )

    def save_preds(self, preds, stage: Stages, dst_path: Path):
        """Save predictions to file.

        Stored as lines of captions

        Args:
            preds (list[list[str]]): Predictions
            stage (Stages): Stage
            dst_path (Path): Destination path
        """

        if stage == Stages.predict:
            if isinstance(preds[0], list):
                preds = sum(preds, [])
            preds = pd.DataFrame({"caption": preds})
            preds.to_csv(dst_path / "captions.txt", index=False)
         

    def train_dataloader(self):
        return DataLoader(
            ImageToTextDataset(
                images_path=self.train_source, 
                df=self.train_df,
                encoder=self.tokenizer_model,
                transforms=self.augmentations['train'],
                captions_length=self.max_caption_length,
            ), batch_size=self.batch_size
        )
