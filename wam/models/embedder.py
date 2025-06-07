# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from ..modules.vae import VAEEncoder, VAEDecoder
from ..modules.msg_processor import MsgProcessor


class Embedder(nn.Module):
    """
    Abstract class for watermark embedding.
    """
    def __init__(self) -> None:
        super(Embedder, self).__init__()
    
    def get_random_msg(self, bsz: int = 1, nb_repetitions = 1) -> torch.Tensor:
        """
        Generate a random message
        """
        return ...

    def get_last_layer(self) -> torch.Tensor:
        return None

    def forward(
        self, 
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        return ...


class VAEEmbedder(Embedder):
    """
    Inserts a watermark into an image.
    """
    def __init__(
        self,
        encoder: VAEEncoder,
        decoder: VAEDecoder,
        msg_processor: MsgProcessor
    ) -> None:
        super(VAEEmbedder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.msg_processor = msg_processor

    def get_random_msg(self, bsz: int = 1, nb_repetitions = 1) -> torch.Tensor:
        return self.msg_processor.get_random_msg(bsz, nb_repetitions)  # b x k

    def get_last_layer(self) -> torch.Tensor:
        last_layer = self.decoder.conv_out.weight
        return last_layer

    def forward(
        self, 
        imgs: torch.Tensor,
        msgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            imgs: (torch.Tensor) Batched images with shape BxCxHxW
            msgs: (torch.Tensor) Batched messages with shape BxL, or empty tensor.
        Returns:
            The watermarked images.
        """
        latents = self.encoder(imgs)
        latents_w = self.msg_processor(latents, msgs)
        imgs_w = self.decoder(latents_w)
        return imgs_w


def build_embedder(name, cfg, nbits):
    if name.startswith('vae'):
        # updates some cfg
        cfg.msg_processor.nbits = nbits
        cfg.msg_processor.hidden_size = nbits * 2
        cfg.decoder.z_channels = (nbits * 2) + cfg.encoder.z_channels
        # build the encoder, decoder and msg processor
        encoder = VAEEncoder(**cfg.encoder)
        msg_processor = MsgProcessor(**cfg.msg_processor)
        decoder = VAEDecoder(**cfg.decoder)
        embedder = VAEEmbedder(encoder, decoder, msg_processor)
    else:
        raise NotImplementedError(f"Model {name} not implemented")
    return embedder
