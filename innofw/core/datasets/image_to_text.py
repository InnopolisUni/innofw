import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import sentencepiece as spm
from torch.utils.data import Dataset


class ImageToTextDataset(Dataset):
        def __init__(
            self,
            images_path: str,
            # captions_path: str,
            df: pd.DataFrame,
            transforms = None,
            encoder: spm.SentencePieceProcessor = None,
            captions_length: int = 128,
        ) -> None:
            super().__init__()
            self.train_source = images_path
            # self.caption_source = captions_path
            self._df = df.copy()
            self.transforms = transforms
            self.encoder = encoder
            self.caption_length = captions_length


        def __len__(self):
            return len(self._df.index)

        def __getitem__(self, index):
            data = self._df.iloc[index]
            image = torch.Tensor(
                np.array(
                    Image.open(
                        os.path.join(self.train_source, "Images", data["image"]),
                    ).convert("RGB")
                ).transpose(2, 0, 1)
            )
            encoded = self.encoder.Encode(data["caption"])
            image = self.transforms(image)

            finished = encoded[: self.caption_length] + [self.encoder.eos_id()]
            true_length = len(finished)

            # Add padding
            padded = finished + [self.encoder.pad_id()] * (self.caption_length - len(finished)) 
            
            return image, torch.LongTensor(padded), true_length