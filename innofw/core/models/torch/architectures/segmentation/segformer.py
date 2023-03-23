#
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import SegformerConfig
from transformers import SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation


#


class SegFormer(nn.Module):
    """
    SegFormer model for segmentation task
    ...

    Attributes
    ----------
    model : nn.Module
        SegFormer model by huggingface

    Methods
    -------
    forward(x):
        returns result of the data forwarding

    """

    def __init__(
        self,
        arch: str = "nvidia/mit-b0",
        retain_dim: bool = False,
        num_channels: Optional[int] = 3,
        num_encoder_blocks: Optional[int] = 4,
        depths: Optional[List[int]] = None,
        sr_ratios: Optional[List[int]] = None,
        hidden_sizes: Optional[List[int]] = None,
        patch_sizes: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        num_attention_heads: Optional[List[int]] = None,
        mlp_ratios: Optional[List[int]] = None,
        hidden_act: Optional[Union[str, Callable]] = "gelu",
        hidden_dropout_prob: Optional[float] = 0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=256,
        is_encoder_decoder=False,
        semantic_loss_ignore_index=255,
        **kwargs,
    ):
        # resolve possible config conflicts
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]
        if sr_ratios is None:
            sr_ratios = [8, 4, 2, 1]
        if hidden_sizes is None:
            hidden_sizes = [32, 64, 160, 256]
        if patch_sizes is None:
            patch_sizes = [7, 3, 3, 3]
        if strides is None:
            strides = [4, 2, 2, 2]
        if num_attention_heads is None:
            num_attention_heads = [1, 2, 5, 8]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]

        encoder_parameters = [depths, sr_ratios, hidden_sizes]

        if not all(
            len(hyper_param) == num_encoder_blocks
            for hyper_param in encoder_parameters
        ):
            raise ValueError(
                "Failed to instantiate model, encoder params must have the same length"
            )

        configuration = SegformerConfig(
            num_channels=num_channels,
            num_encoder_blocks=num_encoder_blocks,
            depths=depths,
            sr_ratios=sr_ratios,
            hidden_sizes=hidden_sizes,
            patch_sizes=patch_sizes,
            strides=strides,
            num_attention_heads=num_attention_heads,
            mlp_ratios=mlp_ratios,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            classifier_dropout_prob=classifier_dropout_prob,
            initializer_range=initializer_range,
            drop_path_rate=drop_path_rate,
            layer_norm_eps=layer_norm_eps,
            decoder_hidden_size=decoder_hidden_size,
            is_encoder_decoder=is_encoder_decoder,
            semantic_loss_ignore_index=semantic_loss_ignore_index,
            **kwargs,
        )
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(
            arch
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            arch
        )  # ,config=configuration)
        self.retain_dim = retain_dim

    def forward(self, x: Union[PIL.Image.Image, np.ndarray, torch.Tensor]):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        out = self.model(**inputs).logits

        if self.retain_dim:
            if isinstance(x, PIL.Image.Image):
                size = x.size[::-1]
            elif isinstance(x, np.ndarray):
                size = x.shape[2:][::-1]
            else:  # x is a torch.Tensor
                size = x.size()[2:][::-1]
            print(size)
            out = nn.functional.interpolate(
                out, size=size, mode="bilinear", align_corners=False
            )
        return out


if __name__ == "__main__":
    seg = SegFormer(
        arch="nvidia/segformer-b0-finetuned-ade-512-512", retain_dim=True
    )

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    print(image.size)

    outputs = seg(image)
    print(outputs.size())

    # feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    # model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    #
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # print(image.size)
    # inputs = feature_extractor(images=image, return_tensors="pt")
    # outputs = model(**inputs)
    # logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
    # print(list(logits.shape))
    # print(inputs.pixel_values.size())
    # mask = nn.functional.interpolate(
    #     logits,
    #     size=image.size[::-1],  # (height, width)
    #     mode='bilinear',
    #     align_corners=False
    # )
    # print(list(mask.shape))
