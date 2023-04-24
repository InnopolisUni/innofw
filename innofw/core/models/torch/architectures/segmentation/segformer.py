#
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
from transformers import SegformerConfig
from transformers import SegformerForSemanticSegmentation


#


class SegFormer(nn.Module):
    """
    SegFormer model for segmentation task
    ...

    Attributes
    ----------
    model : SegformerForSemanticSegmentation
        SegFormer model by https://huggingface.co/nielsr
    retain_dim: bool
        Indicator for upsampling the output to input size

    Methods
    -------
    forward(x):
        returns result of "model" forwarding

    """

    def __init__(
        self,
        arch: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        retain_dim: bool = False,
        num_labels: int = 150,
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
        *args,
        **kwargs,
    ):
        super().__init__()
        hyper_params = {
            "num_channels": num_channels,
            "num_encoder_blocks": num_encoder_blocks,
            "depths": depths,
            "sr_ratios": sr_ratios,
            "hidden_sizes": hidden_sizes,
            "patch_sizes": patch_sizes,
            "strides": strides,
            "num_attention_heads": num_attention_heads,
            "mlp_ratios": mlp_ratios,
            "hidden_act": hidden_act,
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "classifier_dropout_prob": classifier_dropout_prob,
            "initializer_range": initializer_range,
            "drop_path_rate": drop_path_rate,
            "layer_norm_eps": layer_norm_eps,
            "decoder_hidden_size": decoder_hidden_size,
            "is_encoder_decoder": is_encoder_decoder,
            "semantic_loss_ignore_index": semantic_loss_ignore_index,
        }
        hyper_params = {
            name: param for name, param in hyper_params.items() if param
        }

        configuration = SegformerConfig(
            num_labels=num_labels,
            **hyper_params,
        )
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            arch,
            ignore_mismatched_sizes=True,
            config=configuration,
        )
        self.retain_dim = retain_dim

    def forward(self, x: torch.Tensor):
        out = self.model(pixel_values=x).logits
        if self.retain_dim:
            size = tuple(x.shape[2:][::-1])
            out = nn.functional.interpolate(
                out, size=size, mode="bilinear", align_corners=False
            )
        return out
