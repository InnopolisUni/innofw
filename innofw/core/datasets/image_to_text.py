import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
import sentencepiece as spm
from torch.utils.data import Dataset
import logging

class ImageToTextDataset(Dataset):
    """Dataset for image to text task
    
    The dataset is expected to be in the following format:
    ```

    ├── Images
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    |   └── ...
    ├── captions.txt

    ``` 

    The ``captions.txt`` file should be a csv file with the following format:
    ```
    image,caption
    000000.jpg,This is a caption
    000001.jpg,This is another caption
    ...
    ```
    """

    def __init__(
        self,
        images_path: str,
        df: pd.DataFrame,
        transforms = None,
        encoder: spm.SentencePieceProcessor = None,
        captions_length: int = 128,
    ) -> None:
        """Initialize the dataset

        Args: 
            images_path (str): Path to the images folder
            df (pd.DataFrame): Dataframe with the captions
            transforms (Optional[Callable], optional): Transforms to apply to the images. Defaults to None.
            encoder (spm.SentencePieceProcessor, optional): SentencePiece encoder. Defaults to None.
            captions_length (int, optional): Max caption length. Defaults to 128.
        """
        super().__init__()
        self.train_source = images_path
        self._df = df.copy()
        self.transforms = transforms
        self.encoder = encoder
        self.caption_length = captions_length


    def __len__(self):
        return len(self._df.index)

    def __getitem__(self, index):
        """Get an item from the dataset

        Args:
            index (int): Index of the item to get

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: 
                Image, encoded caption and caption length without padding
        """
        data = self._df.iloc[index]
        image = torch.Tensor(
            np.array(
                Image.open(
                    os.path.join(self.train_source, "Images", data["image"]),
                ).convert("RGB")
            ).transpose(2, 0, 1)
        )
        try:
            encoded = self.encoder.Encode(data["caption"])
        except:
            logging.warn(f"Could not encode caption. {type(data['caption'])}. Returning empty caption")
            encoded = []

        image = self.transforms(image)
        finished = encoded[: self.caption_length] + [self.encoder.eos_id()]
        true_length = len(finished)

        # Add padding
        padded = finished + [self.encoder.pad_id()] * (self.caption_length - len(finished)) 
        return image, torch.LongTensor(padded), true_length
