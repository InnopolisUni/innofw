from innofw.core.datamodules.lightning_datamodules.base import BaseLightningDataModule
from innofw.constants import Stages
from pathlib import Path
from typing import Optional, Dict
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import os
import re
import PIL.Image as pil
import torch

class ImageToTextDatamodule(BaseLightningDataModule):
    task = "image-to-text"
    
    @staticmethod
    def split_sentence(sentence):
        return re.findall("\w+", sentence.lower())

    @staticmethod
    def encode_sentence(words, sentence, length: int):
        sentence = ImageToTextDatamodule.split_sentence(sentence)
        sentence = [words.get(x, words.get('<?>')) for x in sentence]
        if len(sentence) + 2 > length:
            raise ValueError(f"Unable to encode sentence of size {len(sentence)}\
                             in a sentence of target length {length}")
        sentence = [words.get("<s>")] + sentence + [words.get('</s>')]
        while len(sentence) < length: sentence.append(words.get('<pad>'))
        return sentence

    class ImageToTextDataset(Dataset):
        def __init__(self, 
                     train_source,
                     df: pd.DataFrame, 
                     word2int: Dict[str, int],
                     max_caption_length=128,
                     ) -> None:
            super().__init__()
            self.df = df
            self.train_source = train_source
            self.length = max_caption_length
            self.word2int = word2int

        def _encode(self, caption):
            return torch.LongTensor(
                ImageToTextDatamodule.encode_sentence(self.word2int, caption, self.length)
            )
        
        def __len__(self): return len(self.df.index)
        
        def __getitem__(self, index):
            data = self.df.iloc[index]
            image = torch.IntTensor(
                    np.array(
                        pil.open(
                            os.path.join(self.train_source, "Images", data['image']),
                        ).convert("RGB")
                    ).transpose(2, 0, 1)
                )
            return self._encode(data['caption']), image
        

    def __init__(self, 
                 train: Optional[Dict[str, str]] = None, 
                 test: Optional[Dict[str, str]] = None, 
                 infer: Optional[Dict[str, str]] = None, 
                 stage: Stages = Stages.train, 
                 word2int: Dict[str, int]=None,
                 max_caption_length=128,
                 *args,
                 **kwargs
                 ):

        super().__init__(train, test, infer=infer, stage=stage, *args, **kwargs)
        self.word2int = word2int
        self.max_caption_length = max_caption_length

    def setup_train_test_val(self):
        self.train_df = pd.read_csv(
            os.path.join(self.train_source, "captions.txt")
        )

        if self.word2int is None:
            self.word2int = dict({'<pad>': 1, '<s>': 2, '</s>': 3, '<?>': 0})
            word_set = set()
            for sentence in self.train_df['caption']:
                word_set.update(ImageToTextDatamodule.split_sentence(sentence))
            self.word2int.update(enumerate(word_set, start=len(self.word2int)))

        return super().setup_train_test_val()
    
    def setup_infer(self):
        return super().setup_infer()

    def save_preds(self, preds, stage: Stages, dst_path: Path):
        return super().save_preds(preds, stage, dst_path)
    
    def test_dataloader(self):
        return super().test_dataloader()
    
    def train_dataloader(self):
        return DataLoader(
            ImageToTextDatamodule.ImageToTextDataset(
                self.train_source,
                self.train_df,
                self.word2int,
                self.max_caption_length
            )
        )
    
    def predict_dataloader(self):
        return super().predict_dataloader()
    