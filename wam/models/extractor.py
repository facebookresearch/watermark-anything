# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from ..modules.vit import ImageEncoderViT
from ..modules.pixel_decoder import PixelDecoder


class Extractor(nn.Module):
    """
    Abstract class for watermark detection.
    """
    def __init__(self) -> None:
        super(Extractor, self).__init__()

    def forward(
        self, 
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            The predicted masks and/or messages.
        """
        return ...


class SegmentationExtractor(Extractor):
    """
    Detects the watermark in an image as a segmentation mask + a message.
    """
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        pixel_decoder: PixelDecoder,
    ) -> None:
        super(SegmentationExtractor, self).__init__()
        self.image_encoder = image_encoder
        self.pixel_decoder = pixel_decoder

    def forward(
        self, 
        imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
        Returns:
            masks: (torch.Tensor) Batched masks with shape Bx(1+nbits)xHxW
        """
        latents = self.image_encoder(imgs)
        masks = self.pixel_decoder(latents)

        return masks


def build_extractor(name, cfg, img_size, nbits):
    if name.startswith('sam'):
        cfg.encoder.img_size = img_size  
        cfg.pixel_decoder.nbits = nbits
        image_encoder = ImageEncoderViT(**cfg.encoder)
        pixel_decoder = PixelDecoder(**cfg.pixel_decoder)
        extractor = SegmentationExtractor(image_encoder=image_encoder, pixel_decoder=pixel_decoder)
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    return extractor